"""Test integration"""

import time
from farms_bullet.network import SalamanderNetwork
import numpy as np
from scipy import integrate

import matplotlib.pyplot as plt


def ode(_time, phases, freqs, coupling_weights, phases_desired, n_dim):
    """Network ODE"""
    phase_repeat = np.repeat(np.array([phases]).T, n_dim, axis=1)
    return freqs + np.sum(
        coupling_weights*np.sin(
            phase_repeat.T-phase_repeat + phases_desired
        ),
        axis=1
    )


def main():
    """Main"""
    timestep = 1e-3
    freqs = 2*np.pi*10*np.ones(11 + 2*2*3)
    times = np.arange(0, 10, 1e-3)

    # Casadi integration
    phases_cas = np.zeros([len(times)+1, len(freqs)])
    network = SalamanderNetwork.from_gait("walking", timestep=timestep)
    tic = time.time()
    for i, _time in enumerate(times):
        freqs = 2*np.pi*10*np.ones(11 + 2*2*3)*(np.sin(_time)+1)
        phases_cas[i+1, :] = network.control_step(freqs)[:, 0]
    print("Casadi integration took {} [s]".format(time.time() - tic))

    # Numpy integration
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

    # Scipy integration
    methods = ["vode", "zvode", "lsoda", "dopri5", "dop853"]
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

    # Plot results
    plt.figure("Casadi")
    plt.plot(times, phases_cas[:-1])
    plt.xlabel("Time [s]")
    plt.ylabel("Phases [rad]")
    plt.grid()

    plt.figure("Numpy")
    plt.plot(times, phases_num[:-1])
    plt.xlabel("Time [s]")
    plt.ylabel("Phases [rad]")
    plt.grid()

    plt.show()


if __name__ == '__main__':
    main()
