"""Test integration"""

import time
from farms_bullet.network import SalamanderNetwork
import numpy as np
from scipy import integrate
import scipy
import sympy as sp
from sympy.utilities.autowrap import autowrap
from sympy.utilities.autowrap import ufuncify
from sympy.utilities.autowrap import binary_function
from sympy import symbols, IndexedBase, Idx, Eq
import matplotlib.pyplot as plt


def ode(_, phases, freqs, coupling_weights, phases_desired, n_dim):
    """Network ODE"""
    phase_repeat = np.repeat(np.expand_dims(phases, axis=1), n_dim, axis=1)
    return freqs + np.sum(
        coupling_weights*np.sin(
            phase_repeat.T-phase_repeat + phases_desired
        ),
        axis=1
    )


def ode_sparse(_, phases, freqs, coupling_weights, phases_desired, n_dim):
    """Network ODE"""
    phase_repeat = np.repeat(np.array([phases]).T, n_dim, axis=1)
    return freqs + coupling_weights.multiply(
        np.sin(phase_repeat.T-phase_repeat + phases_desired)
    ).sum(axis=1).A1


def ode_sym(_, phases, freqs, weights, phases_desired, size, **kwargs):
    """Network ODE"""
    sparse = kwargs.pop("sparse", False)
    weights_num = kwargs.pop("weights_num", None)
    if sparse and weights_num is None:
        raise Exception("weights_num must be provided if sparse")
    # _ode = freqs - sp.Matrix([sum([
    #         (
    #             weights[i, j]
    #             if not sparse or weights_num[i, j]**2 > 1e-3
    #             else 0
    #         )*sp.sin(phases[i] - phases[j] - phases_desired[i, j])
    #         for j in range(size)
    #     ])
    #     for i in range(size)
    # ])
    if sparse:
        weights = [
            [
                weights[i, j]
                if weights_num[i, j]**2 > 1e-3
                else 0
                for j in range(size)
            ]
            for i in range(size)
        ]
    weights = sp.Matrix(weights)
    _phase_matrix = sp.Matrix([
        [
            sp.sin(phases[j] - phases[i] + phases_desired[i, j])
            for j in range(size)
        ]
        for i in range(size)
    ])
    _ode = freqs + sp.dense.matrix_multiply_elementwise(
        weights,
        _phase_matrix
    )*sp.ones(freqs.shape[0], 1)
    return _ode


class System:
    """ODE system for integration"""

    def __init__(self, times, state, ode_fun, *args, jac=None, method="lsoda"):
        super(System, self).__init__()
        self._state = state
        self.timestep = times[1] - times[0]
        self._ode_fun = ode_fun
        self._ode_jac = jac
        if not jac:
            self.ode = integrate.ode(self.ode_fun)  # , jac=jac
        else:
            self.ode = integrate.ode(self.ode_fun, jac=self.ode_jac)
        # self.ode.set_integrator(method)
        self.ode.set_integrator(method, atol=1e-3, rtol=1e-3, nsteps=10)
        self.ode.set_initial_value(self._state, t=0)
        self.ode.set_f_params(*args)
        self.ode.set_jac_params(*args[1:])

    @classmethod
    def from_sympy(cls, times, weights_num, phases_desired_num, **kwargs):
        """ODE from sympy"""
        verbose = kwargs.pop("verbose", False)
        n_dim = 11 + 2*2*3
        phases_num = np.zeros([n_dim, 1])
        freqs_num = kwargs.pop("freqs_num", np.zeros([n_dim, 1]))
        # timestep = times[1] - times[0]
        _time = sp.symbols("t")
        # coupling_weights = sp.SparseMatrix(sp.MatrixSymbol("W", n_dim, n_dim))
        freqs = sp.MatrixSymbol("f", n_dim, 1)
        phases = sp.MatrixSymbol("theta", n_dim, 1)
        dphases = sp.MatrixSymbol("dtheta", n_dim, 1)
        phases_desired = sp.MatrixSymbol("theta_d", n_dim, n_dim)
        # phases_desired_sym = sp.Matrix(phases_desired)
        jacobian = sp.MatrixSymbol("jac", n_dim, n_dim)
        coupling_weights = sp.MatrixSymbol("W", n_dim, n_dim)
        # coupling_weights_sym = sp.Matrix(coupling_weights)
        # coupling_weights_sparse = sp.Matrix(coupling_weights)
        # for i in range(n_dim):
        #     for j in range(n_dim):
        #         coupling_weights_sparse[i, j] = (
        #             coupling_weights[i, j]
        #             if weights_num[i, j]**2 > 1e-6
        #             else 0
        #         )

        # # Expression
        # expr_sym = ode_sym(
        #     _time,
        #     sp.symbols(["theta_{}".format(i) for i in range(n_dim)]),
        #     sp.symbols(["f_{}".format(i) for i in range(n_dim)]),
        #     sp.MatrixSymbol("W", n_dim, n_dim),
        #     sp.MatrixSymbol("theta_d", n_dim, n_dim),
        #     n_dim,
        #     weights_num
        # )
        # print("")
        # for exp in expr_sym:
        #     sp.pretty_print(exp)
        # print("")

        expr = sp.Matrix(ode_sym(
            _time, phases, freqs,
            coupling_weights, phases_desired,
            n_dim, sparse=True, weights_num=weights_num
        ))

        if verbose:
            print("\nODE:\n")
            for exp in expr:
                sp.pretty_print(exp)
            print("")
        # for exp in expr:
        #     sp.pretty_print(exp)
        # expr = Eq(dphases, freqs + sp.Matrix([
        #     sum([
        #         coupling_weights_sparse[i, j]*sp.sin(
        #             phases[i] - phases[j] + phases_desired[i, j]
        #         )
        #         for j in range(n_dim)
        #     ]) for i in range(n_dim)
        # ]))
        # sp.pretty_print(expr)
        # print(" ")

        method = kwargs.pop("method", "autowrap")
        use_jacobian = kwargs.pop("use_jacobian", True)
        backend = kwargs.pop("backend", ["numpy"])  # "math" is faster?
        if verbose:
            print("Using {} method for generating integrator".format(method))
            print("Use jacobian: {}".format(use_jacobian))

        if use_jacobian:
            jac_expr = expr.jacobian(phases)

        # ODE
        if method == "lambdify":
            _ode = sp.lambdify(
                (
                    _time, phases,
                    freqs, coupling_weights, phases_desired
                ),
                expr,
                modules=backend  # "math" is faster?
            )
            _ode_jac = (
                sp.lambdify(
                    (
                        _time, phases,
                        coupling_weights, phases_desired
                    ),
                    jac_expr,
                    modules=backend
                )
                if use_jacobian else None
            )
        elif method == "ufuncify":
            # NOT WORKING
            _ode = ufuncify(
                args=(
                    _time, phases,
                    freqs, coupling_weights, phases_desired
                ),
                expr=expr,
                backend="numpy"
            )
            _ode_jac = None
        elif method == "autowrap":
            _ode = autowrap(
                expr,
                args=(
                    _time, phases, freqs,
                    coupling_weights, phases_desired  # , dphases
                ),
                tempdir="./temp",
                language="C",
                backend="cython",
                verbose=False
            )
            _ode_jac = (
                autowrap(
                    jac_expr,
                    args=(
                        _time, phases,
                        coupling_weights, phases_desired  # , dphases
                    ),
                    tempdir="./temp_jac",
                    language="C",
                    backend="cython",
                    verbose=False
                )
                if use_jacobian
                else None
            )
        else:
            raise Exception("Unknown method for integrator creation")
        return cls(
            times, phases_num, _ode,
            freqs_num, weights_num, phases_desired_num,
            jac=_ode_jac
        )

    def step(self, *args):
        """Step ODE"""
        self.ode.set_f_params(*args)
        self.ode.set_jac_params(*args[1:])
        self._state = self.ode.integrate(self.ode.t+self.timestep)
        return self._state

    def ode_fun(self, _time, _state, freqs, weights, phases_desired):
        """ODE"""
        return self._ode_fun(
            _time, np.expand_dims(_state, 1),
            freqs, weights, phases_desired
        )

    def ode_jac(self, _time, _state, weights, phases_desired):
        """ODE"""
        return self._ode_jac(
            _time, np.expand_dims(_state, 1),
            weights, phases_desired
        )


def test_casadi(times):
    """Casadi"""
    freqs = 2*np.pi*10*np.ones(11 + 2*2*3)
    timestep = times[1] - times[0]
    phases_cas = np.zeros([len(times)+1, len(freqs)])
    network = SalamanderNetwork.from_gait("walking", timestep=timestep)
    tic = time.time()
    for i, _time in enumerate(times):
        freqs = 2*np.pi*10*np.ones(11 + 2*2*3)*(np.sin(_time)+1)
        phases_cas[i+1, :] = network.control_step(freqs)[:, 0]
    print("Casadi integration took {} [s]".format(time.time() - tic))

    # Plot results
    plt.figure("Casadi")
    plt.plot(times, phases_cas[:-1])
    plt.xlabel("Time [s]")
    plt.ylabel("Phases [rad]")
    plt.grid()


def test_numpy_euler(times):
    """Numpy Euler"""
    freqs = 2*np.pi*10*np.ones(11 + 2*2*3)
    timestep = times[1] - times[0]
    n_dim, _phases, _freqs, weights, phases_desired = (
        SalamanderNetwork.walking_parameters()
    )
    phases_num = np.zeros([len(times)+1, len(freqs)])
    phases = np.zeros([len(freqs)])
    tic = time.time()
    for i, _time in enumerate(times):
        phases += timestep*ode(
            _time, phases,
            2*np.pi*10*np.ones(11 + 2*2*3)*(np.sin(_time)+1),
            weights, phases_desired, n_dim
        )
        phases_num[i+1, :] = phases
    print("Numpy integration took {} [s]".format(time.time() - tic))

    # Plot results
    plt.figure("Numpy")
    plt.plot(times, phases_num[:-1])
    plt.xlabel("Time [s]")
    plt.ylabel("Phases [rad]")
    plt.grid()


def test_numpy_euler_sparse(times):
    """Numpy Euler"""
    freqs = 2*np.pi*10*np.ones(11 + 2*2*3)
    timestep = times[1] - times[0]
    n_dim, _phases, _freqs, weights, phases_desired = (
        SalamanderNetwork.walking_parameters()
    )
    phases_num = np.zeros([len(times)+1, len(freqs)])
    phases = np.zeros([len(freqs)])
    weights = scipy.sparse.csr_matrix(weights)
    tic = time.time()
    for i, _time in enumerate(times):
        freqs = 2*np.pi*10*np.ones(11 + 2*2*3)*(np.sin(_time)+1)
        phases += timestep*ode_sparse(
            _time, phases, freqs, weights, phases_desired, n_dim
        )
        phases_num[i+1, :] = phases
    print("Numpy (sparse) integration took {} [s]".format(time.time() - tic))

    # Plot results
    plt.figure("Numpy_sparse")
    plt.plot(times, phases_num[:-1])
    plt.xlabel("Time [s]")
    plt.ylabel("Phases [rad]")
    plt.grid()


def test_scipy(times, methods=None):
    """Scipy"""
    freqs = 2*np.pi*10*np.ones(11 + 2*2*3)
    if not methods:
        methods = ["vode", "zvode", "lsoda", "dopri5", "dop853"]
    timestep = times[1] - times[0]
    n_dim, _phases, _freqs, weights, phases_desired = (
        SalamanderNetwork.walking_parameters()
    )
    for method in methods:
        phases_sci = np.zeros([len(times)+1, len(freqs)])
        phases = np.zeros([len(freqs)])
        r = integrate.ode(ode)
        # r.set_integrator(method)
        r.set_integrator(method, atol=1e-3, rtol=1e-3, nsteps=10)
        r.set_initial_value(phases, 0)
        r.set_f_params(freqs, weights, phases_desired, n_dim)
        tic = time.time()
        for i, _time in enumerate(times):
            r.set_f_params(
                2*np.pi*10*np.ones(11 + 2*2*3)*(np.sin(_time)+1),
                weights,
                phases_desired,
                n_dim
            )
            phases_sci[i+1, :] = r.integrate(r.t+timestep)
        print("Scipy integration took {} [s] with {}".format(
            time.time() - tic,
            method
        ))

        # Plot results
        plt.figure("Scipy ({})".format(method))
        plt.plot(times, phases_sci[:-1])
        plt.xlabel("Time [s]")
        plt.ylabel("Phases [rad]")
        plt.grid()


def test_sympy(times):
    """Test sympy"""
    _, _phases, _freqs, weights, phases_desired = (
        SalamanderNetwork.walking_parameters()
    )
    n_dim = 11 + 2*2*3
    _phases = np.zeros([n_dim, 1], dtype=np.float64)
    sys = System.from_sympy(times, weights, phases_desired)
    phases_sym = np.zeros([len(times)+1, n_dim], dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    phases_desired = np.array(phases_desired, dtype=np.float64)
    tic = time.time()
    for i, _time in enumerate(times):
        phases_sym[i+1, :] = sys.step(
            2*np.pi*10*np.ones([11 + 2*2*3, 1], dtype=np.float64)*(np.sin(_time)+1),
            weights,
            phases_desired
        )[:, 0]
    print("Scipy/Sympy integration took {} [s]".format(
        time.time() - tic
    ))

    # Plot results
    plt.figure("Scipy/sympy")
    plt.plot(times, phases_sym[:-1])
    plt.xlabel("Time [s]")
    plt.ylabel("Phases [rad]")
    plt.grid()


# def test_sympy_tensor():
#     """Test sympy tensor"""
#     from sympy.utilities.autowrap import autowrap
#     from sympy import symbols, IndexedBase, Idx, Eq
#     A, x, y = map(IndexedBase, ['A', 'x', 'y'])
#     m, n = symbols('m n', integer=True)
#     i = Idx('i', m)
#     j = Idx('j', n)
#     instruction = Eq(y[i], A[i, j]*x[j])
#     sp.pretty_print(instruction)


# def test_sympy_compilation():
#     """Test sympy tensor"""
#     freq = sp.MatrixSymbol("f", 2, 1)
#     w = sp.MatrixSymbol("W", 2, 2)
#     phase = sp.MatrixSymbol("theta", 2, 1)
#     dphase = sp.MatrixSymbol("dtheta", 2, 1)
#     phase_d = sp.MatrixSymbol("theta_d", 2, 2)
#     instruction = Eq(dphase, freq + sp.Matrix([
#             sum([
#                 w[i, j]*sp.sin(phase[i] - phase[j] + phase_d[i, j])
#                 for j in range(2)
#             ]) for i in range(2)
#         ])
#     )
#     # instruction = Eq(y[i], A[i, j]*B[i, j]*x[j]*sp.sin(i))
#     # instruction = Eq(
#     #     [dphase[i], phase[i]],
#     #     [
#     #         freq[i] + w[i, j]*sp.sin(phase[i] - phase[j] + phase_d[i, j]),
#     #         phase[i]
#     #     ]
#     #  )
#     sp.pretty_print(instruction)
#     # matvec = autowrap(
#     #     instruction,
#     #     language="C",
#     #     backend="cython",
#     #     tempdir="./temp",
#     #     args=(A, B, x, y)
#     # )
#     matvec = autowrap(
#         instruction,
#         # language="C",
#         # backend="cython",
#         tempdir="./temp",
#         args=(phase, freq, w, phase_d, dphase)
#     )
#     # matvec = binary_function(
#     #     "f",
#     #     A[i, j]*B[i, j]*x[j],
#     #     language="C",
#     #     backend="cython",
#     #     tempdir="./temp",
#     #     args=(A, B, x, y)
#     # )
#     # matvec = ufuncify(
#     #     args=(A, B, x),
#     #     expr=A[i, j]*B[i, j]*x[j],
#     #     language="C",
#     #     backend="cython",
#     #     tempdir="./temp",
#     # )
#     print("FUNCTION READY")
#     np_type = np.float64
#     phase = np.ones([2, 1], dtype=np_type)
#     freq = np.ones([2, 1], dtype=np_type)
#     w = np.ones([2, 2], dtype=np_type)
#     phase_d = np.ones([2, 2], dtype=np_type)
#     tic = time.time()
#     for i in range(1000):
#         pass
#     print("Time: {} [s]".format(time.time() - tic))
#     tic = time.time()
#     for i in range(10000):
#         res = matvec(phase, freq, w, phase_d)
#     print("Time: {} [s]".format(time.time() - tic))
#     sp.pretty_print(res)


# def test_sympy_compilation2():
#     """Test sympy tensor"""
#     from sympy.utilities.autowrap import autowrap
#     from sympy import symbols, IndexedBase, Idx, Eq
#     # freq = sp.IndexedBase("f", shape=(2,))
#     # w = sp.IndexedBase("W", shape=(2, 2))
#     # phase = sp.IndexedBase("theta", shape=(2))
#     # dphase = sp.IndexedBase("dtheta", shape=(2))
#     # phase_d = sp.IndexedBase("theta_d", shape=(2, 2))
#     A = sp.IndexedBase("A", shape=(2, 2))
#     B = sp.IndexedBase("B", shape=(2, 2))
#     x = sp.IndexedBase("x", shape=(2,))
#     y = sp.IndexedBase("y", shape=(2,))
#     # m, n = symbols('m n', integer=True)
#     i = Idx('i', 2)
#     j = Idx('j', 2)
#     instruction = Eq(y[i], (A[i, j]*B[i, j])*x[j]*sp.sin(i))
#     # instruction = Eq(y[i], A[i, j]*B[i, j]*x[j]*sp.sin(i))
#     # instruction = Eq(
#     #     [dphase[i], phase[i]],
#     #     [
#     #         freq[i] + w[i, j]*sp.sin(phase[i] - phase[j] + phase_d[i, j]),
#     #         phase[i]
#     #     ]
#     #  )
#     sp.pretty_print(instruction)
#     # matvec = autowrap(
#     #     instruction,
#     #     language="C",
#     #     backend="cython",
#     #     tempdir="./temp",
#     #     args=(A, B, x, y)
#     # )
#     matvec = autowrap(
#         instruction,
#         language="C",
#         backend="cython",
#         tempdir="./temp",
#         args=(A, B, x, y)
#     )
#     # matvec = binary_function(
#     #     "f",
#     #     A[i, j]*B[i, j]*x[j],
#     #     language="C",
#     #     backend="cython",
#     #     tempdir="./temp",
#     #     args=(A, B, x, y)
#     # )
#     # matvec = ufuncify(
#     #     args=(A, B, x),
#     #     expr=A[i, j]*B[i, j]*x[j],
#     #     language="C",
#     #     backend="cython",
#     #     tempdir="./temp",
#     # )
#     np_type = np.float64
#     M = np.ones([2, 2], dtype=np_type)
#     print("FUNCTION READY")
#     res = np.zeros(2, dtype=np_type)
#     res = matvec(
#         M,
#         M.T,
#         np.ones(2, dtype=np_type)
#     )
#     sp.pretty_print(res)


def main():
    """Main"""
    times = np.arange(0, 0.1, 1e-3)

    test_casadi(times)
    test_numpy_euler(times)
    test_numpy_euler_sparse(times)
    test_scipy(times)
    # test_scipy(times, methods=["lsoda"])
    test_sympy(times)

    plt.show()


if __name__ == '__main__':
    main()
