"""Test integration"""

import time
from farms_bullet.network import SalamanderNetwork
import numpy as np
# import autograd.numpy as np
# from autograd import jacobian
from scipy import integrate
import scipy
import sympy as sp
from sympy.utilities.autowrap import autowrap
from sympy.utilities.autowrap import ufuncify
import matplotlib.pyplot as plt
from farms_bullet.cy_controller import odefun, rk4_ode


def ode(_, phases, freqs, coupling_weights, phases_desired, n_dim):
    """Network ODE"""
    phase_repeat = np.repeat(np.expand_dims(phases, axis=1), n_dim, axis=1)
    return freqs + np.sum(
        coupling_weights*np.sin(
            phase_repeat.T-phase_repeat + phases_desired
        ),
        axis=1
    )


def rk4(fun, timestep, t_n, state, *fun_params):
    """Runge-Kutta step integration"""
    k_1 = timestep*fun(t_n, state, *fun_params)
    k_2 = timestep*fun(t_n+0.5*timestep, state+0.5*k_1, *fun_params)
    k_3 = timestep*fun(t_n+0.5*timestep, state+0.5*k_2, *fun_params)
    k_4 = timestep*fun(t_n+timestep, state+k_3, *fun_params)
    return (k_1+2*k_2+2*k_3+k_4)/6


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
    _ode = freqs - sp.Matrix([
        sum([
            (
                weights[i, j]
                if not sparse or weights_num[i, j]**2 > 1e-3
                else 0
            )*sp.sin(phases[i] - phases[j] - phases_desired[i, j])
            for j in range(size)
        ])
        for i in range(size)
    ])
    return _ode


class System:
    """ODE system for integration"""

    def __init__(self, times, state, ode_fun, method="lsoda", **kwargs):
        super(System, self).__init__()
        self._state = state
        self.timestep = times[1] - times[0]
        self._ode_fun = ode_fun
        self._ode_jac = kwargs.pop("jac", None)
        if self._ode_jac is None:
            self.ode = integrate.ode(self.ode_fun)  # , jac=jac
        else:
            self.ode = integrate.ode(self.ode_fun, jac=self.ode_jac)
        self.ode.set_integrator(method)
        # self.ode.set_integrator(method, atol=1e-3, rtol=1e-3, nsteps=10)
        self.ode.set_initial_value(self._state, t=0)
        self.ode.set_f_params(kwargs.pop("fun_args", None))
        if self._ode_jac is not None:
            self.ode.set_jac_params(kwargs.pop("jac_args", None))

    @classmethod
    def from_sympy(cls, times, weights_num, phases_desired_num, **kwargs):
        """ODE from sympy"""
        n_dim = 11 + 2*2*3
        _time = sp.symbols("t")
        phases = sp.MatrixSymbol("theta", n_dim, 1)
        ode_params = (
            sp.MatrixSymbol("f", n_dim, 1),
            sp.MatrixSymbol("W", n_dim, n_dim),
            sp.MatrixSymbol("theta_d", n_dim, n_dim)
        )

        expr = sp.Matrix(ode_sym(
            _time, phases, *ode_params,
            size=n_dim, sparse=True, weights_num=weights_num
        ))

        _ode, _ode_jac = cls.generate_ode(
            expr=expr,
            fun_args=(_time, phases, *ode_params),
            jac_args=(_time, phases, *ode_params[1:]),
            dx=phases,
            **kwargs
        )

        return cls(
            times=times,
            state=np.zeros([n_dim, 1]),
            ode_fun=_ode,
            fun_args=(
                kwargs.pop("freqs_num", np.zeros([n_dim, 1])),
                weights_num,
                phases_desired_num
            ),
            jac=_ode_jac,
            jac_args=(weights_num, phases_desired_num)
        )

    @staticmethod
    def generate_ode(expr, **kwargs):
        """Generate ODE integrator"""
        verbose = kwargs.pop("verbose", False)
        method = kwargs.pop("method", "autowrap")
        use_jacobian = kwargs.pop("use_jacobian", True)
        backend = kwargs.pop("backend", ["numpy"])  # "math" is faster?
        fun_args = kwargs.pop("fun_args", None)
        jac_args = kwargs.pop("jac_args", None)
        if verbose:
            print("\nODE:\n")
            for exp in expr:
                sp.pretty_print(exp)
            print("\nUsing {} method for generating integrator".format(method))
            print("Use jacobian: {}\n".format(use_jacobian))

        if use_jacobian:
            jac_expr = expr.jacobian(kwargs.pop("dx", None))

        # ODE
        if method == "lambdify":
            _ode = sp.lambdify(
                fun_args,
                sp.cse(expr),
                modules=backend
            )
            _ode_jac = (
                sp.lambdify(
                    jac_args,
                    sp.cse(jac_expr),
                    modules=backend
                )
                if use_jacobian else None
            )
        elif method == "ufuncify":
            # NOT WORKING
            _ode = ufuncify(
                args=fun_args,
                expr=expr,
                backend="numpy"
            )
            _ode_jac = None
        elif method == "autowrap":
            _ode = autowrap(
                expr,
                args=fun_args,
                tempdir="./temp",
                # language="C",
                # backend="cython",
                verbose=False
            )
            _ode_jac = (
                autowrap(
                    jac_expr,
                    args=jac_args,
                    tempdir="./temp_jac",
                    # language="C",
                    # backend="cython",
                    verbose=False
                )
                if use_jacobian
                else None
            )
        else:
            raise Exception("Unknown method for integrator creation")
        return _ode, _ode_jac

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
    print("Numpy/Euler integration took {} [s]".format(time.time() - tic))

    # Plot results
    plt.figure("Numpy")
    plt.plot(times, phases_num[:-1])
    plt.xlabel("Time [s]")
    plt.ylabel("Phases [rad]")
    plt.grid()


def test_numpy_rk(times):
    """Numpy Euler"""
    dtype = np.float64
    freqs = 2*np.pi*10*np.ones(11 + 2*2*3)
    timestep = times[1] - times[0]
    n_dim, _phases, _freqs, weights, phases_desired = (
        SalamanderNetwork.walking_parameters()
    )
    phases_num = np.zeros([len(times)+1, len(freqs)], dtype=dtype)
    phases = np.zeros([len(freqs)], dtype=dtype)
    tic = time.time()
    for i, _time in enumerate(times):
        phases += rk4(
            ode, timestep,
            0, phases,
            2*np.pi*10*np.ones(11 + 2*2*3, dtype=dtype)*(np.sin(_time)+1),
            weights, phases_desired, n_dim
        )
        phases_num[i+1, :] = phases
    print("RK4 integration took {} [s]".format(time.time() - tic))

    # Plot results
    plt.figure("RK4")
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


def test_scipy_ode(times, methods=None):
    """Scipy"""
    freqs = 2*np.pi*10*np.ones(11 + 2*2*3)
    if not methods:
        methods = ["vode", "zvode", "lsoda", "dopri5", "dop853"]
    timestep = times[1] - times[0]
    n_dim, _, _, weights, phases_desired = (
        SalamanderNetwork.walking_parameters()
    )
    for method in methods:
        phases_sci = np.zeros([len(times)+1, len(freqs)])
        phases = np.zeros([len(freqs)])
        # jac = jacobian(ode, 1)
        # print(jac(0, phases, freqs, weights, phases_desired, n_dim))
        _ode = integrate.ode(ode)  # , jac=jac
        _ode.set_integrator(method)
        # _ode.set_integrator(method, atol=1e-3, rtol=1e-3, nsteps=10)
        _ode.set_initial_value(phases, 0)
        _ode.set_f_params(freqs, weights, phases_desired, n_dim)
        # _ode.set_jac_params(freqs, weights, phases_desired, n_dim)
        tic = time.time()
        for i, _time in enumerate(times):
            freqs = 2*np.pi*10*np.ones(11 + 2*2*3)*(np.sin(_time)+1)
            _ode.set_f_params(
                freqs,
                weights,
                phases_desired,
                n_dim
            )
            # _ode.set_jac_params(
            #     freqs,
            #     weights,
            #     phases_desired,
            #     n_dim
            # )
            phases_sci[i+1, :] = _ode.integrate(_ode.t+timestep)  # , step=True
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


class ODEFunction:
    """ODE function"""

    def __init__(self, ode_fun, args):
        super(ODEFunction, self).__init__()
        self.ode = ode_fun
        self.args = args

    def set_args(self, args):
        """Set function arguments"""
        self.args = args

    def fun(self, current_time, state):
        """ODE function"""
        return self.ode(current_time, state, *self.args)


def test_scipy_new(times, methods=None):
    """Scipy"""
    freqs = 2*np.pi*10*np.ones(11 + 2*2*3)
    if not methods:
        methods = ["vode", "zvode", "lsoda", "dopri5", "dop853"]
    timestep = times[1] - times[0]
    n_dim, _, _, weights, phases_desired = (
        SalamanderNetwork.walking_parameters()
    )
    for method in methods:
        phases_sci = np.zeros([len(times)+1, len(freqs)])
        phases = np.zeros([len(freqs)])
        _ode_fun = ODEFunction(ode, (freqs, weights, phases_desired, n_dim))
        _ode = method(_ode_fun.fun, 0, phases, 1e3)
        tic = time.time()
        for i, _time in enumerate(times):
            freqs = 2*np.pi*10*np.ones(11 + 2*2*3)*(np.sin(_time)+1)
            _ode_fun.set_args((
                2*np.pi*10*np.ones(11 + 2*2*3)*(np.sin(_time)+1),
                weights,
                phases_desired,
                n_dim
            ))
            _ode.step()
            print(_ode.t)
            phases_sci[i+1, :] = _ode.y
            # _ode.set_jac_params(
            #     freqs,
            #     weights,
            #     phases_desired,
            #     n_dim
            # )
            # phases_sci[i+1, :] = _ode.integrate(_ode.t+timestep)
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


def test_scipy_odeint(times):
    """Scipy"""
    freqs = 2*np.pi*10*np.ones(11 + 2*2*3)
    timestep = times[1] - times[0]
    n_dim, _, _, weights, phases_desired = (
        SalamanderNetwork.walking_parameters()
    )
    phases_sci = np.zeros([len(times)+1, len(freqs)])
    phases = np.zeros([len(freqs)])
    tic = time.time()
    for i, _time in enumerate(times):
        freqs = 2*np.pi*10*np.ones(11 + 2*2*3)*(np.sin(_time)+1)
        _res, _ = scipy.integrate.odeint(
            func=ode,
            y0=phases,
            t=[0, timestep],
            args=(
                2*np.pi*10*np.ones(11 + 2*2*3)*(np.sin(_time)+1),
                weights,
                phases_desired,
                n_dim
            ),
            tfirst=True
        )
        phases_sci[i+1, :] = _res
    print("Scipy integration took {} [s] with odeint".format(
        time.time() - tic
    ))

    # Plot results
    plt.figure("Scipy (odeint)")
    plt.plot(times, phases_sci[:-1])
    plt.xlabel("Time [s]")
    plt.ylabel("Phases [rad]")
    plt.grid()


def test_sympy(times, methods=None):
    """Test sympy"""
    if not methods:
        methods = ["vode", "lsoda", "dopri5", "dop853"]
    _, _phases, _freqs, weights, phases_desired = (
        SalamanderNetwork.walking_parameters()
    )
    n_dim = 11 + 2*2*3
    _phases = np.zeros([n_dim, 1], dtype=np.float64)
    sys = System.from_sympy(times, weights, phases_desired)
    phases_sym = np.zeros([len(times)+1, n_dim], dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    phases_desired = np.array(phases_desired, dtype=np.float64)
    for method in methods:
        sys.ode.set_initial_value(_phases, t=0)
        # sys.ode.set_integrator(method, atol=1e-3, rtol=1e-3, nsteps=10)
        tic = time.time()
        for i, _time in enumerate(times):
            phases_sym[i+1, :] = sys.step(
                2*np.pi*10*np.ones(
                    [11 + 2*2*3, 1],
                    dtype=np.float64
                )*(np.sin(_time)+1),
                weights,
                phases_desired
            )[:, 0]
        print("Scipy/Sympy integration took {} [s] with {}".format(
            time.time() - tic,
            method
        ))

        # Plot results
        plt.figure("Scipy/sympy with {}".format(method))
        plt.plot(times, phases_sym[:-1])
        plt.xlabel("Time [s]")
        plt.ylabel("Phases [rad]")
        plt.grid()


def test_cython(times):
    """Test Cython integration"""
    _, __, __, weights, phases_desired = (
        SalamanderNetwork.walking_parameters()
    )
    dtype = np.float64
    n_dim = 11 + 2*2*3
    timestep = times[1] - times[0]
    state = np.zeros([n_dim], dtype=dtype)
    weights = np.array(weights, dtype=dtype)
    phi = np.array(phases_desired, dtype=dtype)
    phases_cy = np.zeros([len(times)+1, n_dim], dtype=dtype)

    tic = time.time()
    phases_cy[0, :] = state
    for i, _time in enumerate(times):
        rk4_ode(
            odefun,
            timestep,
            state,
            2*np.pi*10*np.ones(
                n_dim,
                dtype=dtype
            )*(np.sin(_time)+1),
            weights,
            phi,
            n_dim
        )
        phases_cy[i+1, :] = state
    print("Cython/RK4 integration took {} [s]".format(time.time() - tic))

    # Plot results
    plt.figure("Cython")
    plt.plot(times, phases_cy[:-1])
    plt.xlabel("Time [s]")
    plt.ylabel("Phases [rad]")
    plt.grid()


def main():
    """Main"""
    times = np.arange(0, 10, 1e-3)

    # test_casadi(times)
    test_numpy_euler(times)
    test_numpy_rk(times)
    test_cython(times)
    # test_numpy_euler_sparse(times)
    # test_scipy_ode(times)
    # test_scipy_ode(times, methods=["lsoda"])
    # test_scipy_new(times, methods=[scipy.integrate.LSODA])
    # test_scipy_odeint(times)
    # test_sympy(times)

    plt.show()


if __name__ == '__main__':
    main()
