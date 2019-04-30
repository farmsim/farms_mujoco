"""Test integration"""

import time
from farms_bullet.controllers.casadi import SalamanderCasADiNetwork
import numpy as np
# import autograd.numpy as np
# from autograd import jacobian
from scipy import integrate
import scipy
import sympy as sp
from sympy.utilities.autowrap import autowrap
from sympy.utilities.autowrap import ufuncify
import matplotlib.pyplot as plt
from farms_bullet.cy_controller_old import (
    odefun, rk4_ode,
    odefun_sparse, rk4_ode_sparse
)
from farms_bullet.controllers.network import SalamanderNetworkODE
from farms_bullet.cy_controller import rk4 as cyrk4, euler as cyeuler


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
    network = SalamanderCasADiNetwork.from_gait("walking", timestep=timestep)
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
        SalamanderCasADiNetwork.walking_parameters()
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
        SalamanderCasADiNetwork.walking_parameters()
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
        SalamanderCasADiNetwork.walking_parameters()
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
        SalamanderCasADiNetwork.walking_parameters()
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



def test_scipy_new(times, methods):
    """Scipy"""
    freqs = 2*np.pi*10*np.ones(11 + 2*2*3)
    timestep = times[1] - times[0]
    n_dim, _, _, weights, phases_desired = (
        SalamanderCasADiNetwork.walking_parameters()
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
            phases_sci[i+1, :] = _ode.y
            # _ode.set_jac_params(
            #     freqs,
            #     weights,
            #     phases_desired,
            #     n_dim
            # )
            # phases_sci[i+1, :] = _ode.integrate(_ode.t+timestep)
        print("Scipy_new integration took {} [s] with {}".format(
            time.time() - tic,
            method
        ))

        # Plot results
        plt.figure("Scipy_new ({})".format(method))
        plt.plot(times, phases_sci[:-1])
        plt.xlabel("Time [s]")
        plt.ylabel("Phases [rad]")
        plt.grid()


def test_scipy_odeint(times):
    """Scipy"""
    freqs = 2*np.pi*10*np.ones(11 + 2*2*3)
    timestep = times[1] - times[0]
    n_dim, _, _, weights, phases_desired = (
        SalamanderCasADiNetwork.walking_parameters()
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
        SalamanderCasADiNetwork.walking_parameters()
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
        SalamanderCasADiNetwork.walking_parameters()
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
    plt.grid(True)


def test_cython_sparse(times):
    """Test Cython integration"""
    _, __, __, weights, phases_desired = (
        SalamanderCasADiNetwork.walking_parameters()
    )
    dtype = np.float64
    n_dim = 11 + 2*2*3
    timestep = times[1] - times[0]
    state = np.zeros([n_dim], dtype=dtype)
    weights = np.array(weights, dtype=dtype)
    phi = np.array(phases_desired, dtype=dtype)
    phases_cy = np.zeros([len(times)+1, n_dim], dtype=dtype)
    connectivity = []
    connections = []
    for i in range(n_dim):
        for j in range(n_dim):
            if weights[i, j]**2 > 1e-6:
                connectivity.append([i, j])
                connections.append([weights[i, j], phi[i, j]])
    connectivity = np.array(connectivity, dtype=np.uintc)
    connections = np.array(connections, dtype=dtype)
    c_dim = np.shape(connectivity)[0]

    tic = time.time()
    phases_cy[0, :] = state
    for i, _time in enumerate(times):
        rk4_ode_sparse(
            odefun_sparse,
            timestep,
            state,
            2*np.pi*10*np.ones(
                n_dim,
                dtype=dtype
            )*(np.sin(_time)+1),
            connectivity,
            connections,
            n_dim,
            c_dim
        )
        phases_cy[i+1, :] = state
    print("Cython/RK4 integration took {} [s]".format(time.time() - tic))

    # Plot results
    plt.figure("Cython sparse")
    plt.plot(times, phases_cy[:-1])
    plt.xlabel("Time [s]")
    plt.ylabel("Phases [rad]")
    plt.grid(True)


def freqs_function(_time, n_dim, freq=3):
    """Frequency function"""
    return 2*np.pi*0.1*np.ones(n_dim)*(np.sin(2*np.pi*freq*_time))


def test_cython_new(times):
    # Allocation
    timestep = times[1] - times[0]

    for method_name, method in [["RK4", cyrk4], ["Euler", cyeuler]]:
        network = SalamanderNetworkODE.walking(
            n_iterations=len(times),
            timestep=timestep
        )
        network.ode["solver"] = method
        n_dim = np.shape(network.phases)[1]

        # Simulate (method 1)
        time_control = 0
        for _time in times[:-1]:
            tic0 = time.time()
            network.parameters.oscillators.freqs = freqs_function(_time, n_dim)
            network.control_step()
            tic1 = time.time()
            time_control += tic1 - tic0
        print("Cython ({}) integration took {} [s]".format(
            method_name,
            time_control
        ))

        # Plot results
        plt.figure("Cython new ({})".format(method_name))
        plt.plot(times, network.phases)
        plt.xlabel("Time [s]")
        plt.ylabel("Phases [rad]")
        plt.grid(True)


class NetworkODEwrap:
    """ODE function"""

    def __init__(self, n_iterations, timestep):
        super(NetworkODEwrap, self).__init__()
        self.network = SalamanderNetworkODE.walking(
            n_iterations=n_iterations,
            timestep=timestep
        )
        self.dstate = np.copy(self.network.state.array[0, 0])
        self.parameters = self.network.parameters.to_ode_parameters().function
        self.i = 0
        self.tot_time = 0

    def fun(self, _time, state):
        """ODE function"""
        self.i += 1
        tic = time.time()
        self.network.ode.function(
            self.dstate,
            state,
            *self.parameters
        )
        self.tot_time += time.time() - tic
        return self.dstate


def test_scipy_cython_ivp(times, methods=None):
    # Allocation
    timestep = times[1] - times[0]

    # Simulate (method 1)
    if not methods:
        methods = ["RK45", "RK23", "LSODA"]  # "BDF", "Radau",
    for method in methods:
        time_control = 0
        network_ode = NetworkODEwrap(len(times), timestep)
        n_dim = np.shape(network_ode.network.parameters.oscillators.freqs)[0]
        for i, _time in enumerate(times[:-1]):
            tic0 = time.time()
            network_ode.network.parameters.oscillators.freqs = (
                freqs_function(_time, n_dim)
            )
            sol = integrate.solve_ivp(
                fun=network_ode.fun,
                t_span=[0, timestep],
                y0=network_ode.network.state.array[i, 0],
                method=method,
                t_eval=[timestep],
                # first_step=timestep,
                # min_step=timestep,
                # max_step=timestep,
                # rtol=1e-3,
                # atol=1e-3
            )
            network_ode.network.state.array[i+1, 0, :] = sol.y[:, -1]
            tic1 = time.time()
            time_control += tic1 - tic0
        print("Number of iterations: {} (t={}[s])".format(
            network_ode.i,
            network_ode.tot_time
        ))
        print("Scipy/Cython (solve_ivp/{}) integration took {} [s]".format(
            method,
            time_control
        ))

        # Plot results
        plt.figure("Scipy/Cython (solve_ivp/{})".format(method))
        plt.plot(times, network_ode.network.phases)
        plt.xlabel("Time [s]")
        plt.ylabel("Phases [rad]")
        plt.grid(True)


def test_scipy_cython_odesolver(times):
    # Allocation
    timestep = times[1] - times[0]

    # Simulate (method 1)
    time_control = 0
    network_ode = NetworkODEwrap(len(times), timestep)
    n_dim = np.shape(network_ode.network.parameters.oscillators.freqs)[0]
    sol = integrate.RK45(
        fun=network_ode.fun,
        t0=0,
        y0=np.copy(network_ode.network.state.array[0, 0]),
        t_bound=1e4,
        # first_step=timestep,
        # min_step=0,
        max_step=timestep,
        rtol=1e-6,
        atol=1e-8
    )
    for i, _time in enumerate(times[:-1]):
        tic0 = time.time()
        network_ode.network.parameters.oscillators.freqs = (
            freqs_function(_time, n_dim)
        )
        sol.step()
        network_ode.network.state.array[i+1, 0, :] = sol.y
        tic1 = time.time()
        time_control += tic1 - tic0
    print("Time: {} [s]".format(sol.t))
    print("Number of iterations: {} (t={}[s])".format(
        network_ode.i,
        network_ode.tot_time
    ))
    print("Scipy/Cython (odesolver) integration took {} [s]".format(
        time_control
    ))

    # Plot results
    plt.figure("Scipy/Cython odesolver")
    plt.plot(times, network_ode.network.phases)
    plt.xlabel("Time [s]")
    plt.ylabel("Phases [rad]")
    plt.grid(True)


def test_scipy_cython_odeint(times):
    # Allocation
    timestep = times[1] - times[0]

    # Simulate (method 1)
    time_control = 0
    network_ode = NetworkODEwrap(len(times), timestep)
    n_dim = np.shape(network_ode.network.parameters.oscillators.freqs)[0]
    for i, _time in enumerate(times[:-1]):
        tic0 = time.time()
        network_ode.network.parameters.oscillators.freqs = (
            freqs_function(_time, n_dim)
        )
        network_ode.network.state.array[i+1, 0, :] = integrate.odeint(
            func=network_ode.fun,
            y0=network_ode.network.state.array[i, 0],
            t=[0, timestep],
            tfirst=True
        )[-1]
        tic1 = time.time()
        time_control += tic1 - tic0
    print("Number of iterations: {} (t={}[s])".format(
        network_ode.i,
        network_ode.tot_time
    ))
    print("Scipy/Cython (odeint) integration took {} [s]".format(
        time_control
    ))

    # Plot results
    plt.figure("Scipy/Cython (odeint)")
    plt.plot(times, network_ode.network.phases)
    plt.xlabel("Time [s]")
    plt.ylabel("Phases [rad]")
    plt.grid(True)


def test_scipy_cython_ode(times, methods=None):
    # Allocation
    timestep = times[1] - times[0]

    # Simulate (method 1)
    if not methods:
        methods = ["vode", "lsoda", "dopri5", "dop853"]
    for method in methods:
        network_ode = NetworkODEwrap(len(times), timestep)
        n_dim = np.shape(network_ode.network.parameters.oscillators.freqs)[0]
        solver = integrate.ode(f=network_ode.fun)
        # , max_step=timestep, first_step=timestep
        solver.set_integrator(method)
        solver.set_initial_value(network_ode.network.state.array[0, 0, :], 0)
        time_control = 0
        for i, _time in enumerate(times[:-1]):
            tic0 = time.time()
            network_ode.network.parameters.oscillators.freqs = (
                freqs_function(_time, n_dim)
            )
            solver.integrate(solver.t+timestep)
            assert (solver.t - (_time+timestep))**2 < 1e-5
            network_ode.network.state.array[i+1, 0, :] = solver.y
            tic1 = time.time()
            time_control += tic1 - tic0
        print("Number of iterations: {} (t={}[s])".format(
            network_ode.i,
            network_ode.tot_time
        ))
        print("Scipy/Cython (ode/{}) integration took {} [s]".format(
            method,
            time_control
        ))

        # Plot results
        plt.figure("Scipy/Cython (ode/{})".format(method))
        plt.plot(times, network_ode.network.phases)
        plt.xlabel("Time [s]")
        plt.ylabel("Phases [rad]")
        plt.grid(True)


def main():
    """Main"""

    # # Old implementation
    # times = np.arange(0, 10, 1e-3)
    # test_casadi(times)
    # test_numpy_euler(times)
    # test_numpy_rk(times)
    # test_numpy_euler_sparse(times)
    # test_scipy_ode(times)
    # # test_scipy_ode(times, methods=["lsoda"])
    # test_scipy_new(times, methods=[scipy.integrate.LSODA])
    # # test_scipy_odeint(times)
    # test_sympy(times)
    # test_cython(times)
    # test_cython_sparse(times)

    # New implementation
    max_time = 1
    timestep = 1e-2
    times = np.arange(0, max_time, timestep)
    print("Times:\ntimes = np.arange(0, {}, {})".format(max_time, timestep))
    print("\nNew Cython:")
    test_cython_new(times)
    print("\nSolve_ivp:")
    test_scipy_cython_ivp(times)
    print("\nODEsolver:")
    test_scipy_cython_odesolver(times)
    print("\nOdeint:")
    test_scipy_cython_odeint(times)
    print("\nODE:")
    test_scipy_cython_ode(times)

    plt.show()


if __name__ == '__main__':
    main()
