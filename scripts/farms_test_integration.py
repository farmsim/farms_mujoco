"""Test integration"""

import time
from farms_bullet.network import SalamanderNetwork
import numpy as np
from scipy import integrate
import scipy
import sympy as sp
from sympy.utilities.autowrap import ufuncify
import matplotlib.pyplot as plt


def ode(_, phases, freqs, coupling_weights, phases_desired, n_dim):
    """Network ODE"""
    phase_repeat = np.repeat(np.array([phases]).T, n_dim, axis=1)
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


def ode2(_, phases, freqs, weights, phases_desired):
    """Network ODE"""
    size = len(phases)
    _ode = [
        freqs[i] + sum([
            weights[i, j]*sp.sin(phases[j] - phases[i] + phases_desired[i, j])
            for j in range(size)
        ])
        for i in range(size)
    ]
    return _ode


class System:
    """ODE system for integration"""

    def __init__(self, times, state, ode_fun, *args, jac=None, method="lsoda"):
        super(System, self).__init__()
        self._state = state
        self.timestep = times[1] - times[0]
        self.ode_fun = ode_fun
        self.ode = integrate.ode(ode_fun, jac=jac)
        self.ode.set_integrator(method, atol=1e-3, rtol=1e-3, nsteps=10)
        self.ode.set_initial_value(self._state, 0)
        self.ode.set_f_params(*args)

    @classmethod
    def from_sympy(cls, times, freqs_num, weights_num, phases_desired_num):
        """ODE from sympy"""
        n_dim = 11 + 2*2*3
        phases_num = np.zeros(n_dim)
        timestep = times[1] - times[0]
        _time = sp.symbols("t")
        phases = sp.symbols(["theta_{}".format(i) for i in range(n_dim)])
        freqs = sp.symbols(["f_{}".format(i) for i in range(n_dim)])
        coupling_weights = np.array([
            [
                sp.symbols("w_{}_{}".format(i, j))
                # if weights_num[i][j]**2 > 1e-3 else 0
                for j, _ in enumerate(phases)
            ] for i, _ in enumerate(phases)
        ])
        coupling_weights_sym = np.array([
            [
                sp.symbols("w_{}_{}".format(i, j))
                if weights_num[i][j]**2 > 1e-3 else 0
                for j, _ in enumerate(phases)
            ] for i, _ in enumerate(phases)
        ])
        phases_desired = np.array([
            [
                sp.symbols("theta_d_{}_{}".format(i, j))
                for j, _ in enumerate(phases)
            ] for i, _ in enumerate(phases)
        ])
        phases_desired_sym = np.array([
            [
                sp.symbols("theta_d_{}_{}".format(i, j))
                for j, _ in enumerate(phases)
            ] for i, _ in enumerate(phases)
        ])

        # freqs_mat = sp.Matrix([[freq] for freq in freqs])

        # Expression
        # expr = ode(_time, phases, freqs, coupling_weights, phases_desired, n_dim)
        # print("\nODE:\n")
        # sp.pretty_print(expr[0])
        # print(" ")

        # Expression
        expr = ode2(
            _time, phases, freqs, coupling_weights_sym, phases_desired_sym
        )
        # jac = [
        #     [
        #         sp.diff(exp, phase)
        #         for phase in phases
        #     ] for exp in expr
        # ]
        print("\nODE:\n")
        for exp in expr:
            sp.pretty_print(exp)
        print(" ")

        # ODE
        _ode = sp.lambdify(
            (_time, phases, freqs, coupling_weights, phases_desired),
            expr,
            "numpy"  # "math" is faster
        )
        # _ode = ufuncify(
        #     (_time, phases, freqs, coupling_weights, phases_desired),
        #     expr
        # )
        # result = _ode(0, phases, freqs, coupling_weights, phases_desired)
        # print("\nODE:\n")
        # sp.pretty_print(result[0])
        # print(" ")
        return cls(
            times, phases_num, _ode,
            freqs_num, weights_num, phases_desired_num,
            jac=None
        )

    def step(self, *args):
        """Step ODE"""
        self.ode.set_f_params(*args)
        self._state = self.ode.integrate(self.ode.t+self.timestep)
        return self._state


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
        freqs = 2*np.pi*10*np.ones(11 + 2*2*3)*(np.sin(_time)+1)
        phases += timestep*ode(
            _time, phases, freqs, weights, phases_desired, n_dim
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
    methods = (
        methods
        if methods
        else ["vode", "zvode", "lsoda", "dopri5", "dop853"]
    )
    timestep = times[1] - times[0]
    n_dim, _phases, _freqs, weights, phases_desired = (
        SalamanderNetwork.walking_parameters()
    )
    for method in methods:
        phases_sci = np.zeros([len(times)+1, len(freqs)])
        phases = np.zeros([len(freqs)])
        r = integrate.ode(ode)
        r.set_integrator(method, atol=1e-3, rtol=1e-3, nsteps=10)
        r.set_initial_value(phases, 0)
        r.set_f_params(freqs, weights, phases_desired, n_dim)
        tic = time.time()
        for i, _time in enumerate(times):
            freqs = 2*np.pi*10*np.ones(11 + 2*2*3)*(np.sin(_time)+1)
            r.set_f_params(freqs, weights, phases_desired, n_dim)
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
    n_dim, _phases, _freqs, weights, phases_desired = (
        SalamanderNetwork.walking_parameters()
    )
    freqs = 2*np.pi*10*np.ones(11 + 2*2*3)
    sys = System.from_sympy(times, freqs, weights, phases_desired)
    a = sys.ode_fun(0, _phases, freqs, weights, phases_desired)
    _state = sys.step(freqs, weights, phases_desired)

    phases_sym = np.zeros([len(times)+1, len(freqs)])
    tic = time.time()
    for i, _time in enumerate(times):
        phases_sym[i+1, :] = sys.step(
            2*np.pi*10*np.ones(11 + 2*2*3)*(np.sin(_time)+1),
            weights,
            phases_desired
        )
    print("Scipy/Sympy integration took {} [s]".format(
        time.time() - tic
    ))

    # Plot results
    plt.figure("Scipy/sympy")
    plt.plot(times, phases_sym[:-1])
    plt.xlabel("Time [s]")
    plt.ylabel("Phases [rad]")
    plt.grid()


def main():
    """Main"""
    times = np.arange(0, 10, 1e-3)

    # test_casadi(times)
    test_numpy_euler(times)
    # test_numpy_euler_sparse(times)
    # test_scipy(times, methods=["lsoda"])
    test_sympy(times)

    plt.show()


if __name__ == '__main__':
    main()
