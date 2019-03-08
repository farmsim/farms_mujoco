"""Compute phase difference"""

import time

import numpy as np
import casadi as cas

import matplotlib.pyplot as plt


def main():
    """Main"""
    timestep = 1e-3
    n_dim = 10
    freqs = np.array([
        cas.SX.sym("freqs_{}".format(i))
        for i in range(n_dim)
    ])
    phases = np.array([
        [
            cas.SX.sym("phases_{}".format(i))
            if not i % 1 else 0
        ]
        for i in range(n_dim)
    ])
    coupling_weights_vals = np.zeros([n_dim, n_dim])
    phases_desired_vals = np.zeros([n_dim, n_dim])
    for i in range(n_dim-1):
        coupling_weights_vals[i, i+1] = 3e2
        coupling_weights_vals[i+1, i] = 3e2
        phases_desired_vals[i, i+1] = -0.1
        phases_desired_vals[i+1, i] = 0.1
    for i in range(n_dim-2):
        coupling_weights_vals[i, i+2] = 0
        coupling_weights_vals[i+2, i] = 0
        phases_desired_vals[i, i+2] = 0
        phases_desired_vals[i+2, i] = 0
    coupling_weights = np.array([
        [
            cas.SX.sym("w_{}_{}".format(i, j))
            if coupling_weights_vals[i, j] != 0
            else 0  # cas.SX.sym("0")
            for j in range(n_dim)
        ] for i in range(n_dim)
    ])
    phases_desired = np.array([
        [
            cas.SX.sym("theta_d_{}_{}".format(i, j))
            if coupling_weights_vals[i, j] != 0
            else 0  # cas.SX.sym("0")
            for j in range(n_dim)
        ] for i in range(n_dim)
    ])
    print("phases:\n{}".format(phases))
    phase_repeat = np.repeat(phases, n_dim, axis=1)
    print("phases_repeat:\n{}".format(phase_repeat))
    phase_diff = phase_repeat.T-phase_repeat
    print("phases_diff:\n{}".format(phase_diff))
    ode = (
        freqs
        + np.sum(
            coupling_weights*np.sin(phase_diff+phases_desired),
            axis=1
        )
    )
    print("ODE:\n{}".format(ode))

    print("Phases:\n{}".format(phases.T))
    print("Freqs:\n{}".format(freqs))
    # print("Coupling weights:\n{}".format(coupling_weights))
    coupling_weights_sym = np.array([
        coupling_weights[i, j]
        for i in range(n_dim)
        for j in range(n_dim)
        if isinstance(coupling_weights[i, j], cas.SX)
    ])
    phases_desired_sym = np.array([
        phases_desired[i, j]
        for i in range(n_dim)
        for j in range(n_dim)
        if isinstance(coupling_weights[i, j], cas.SX)
    ])
    print("Coupling weights sym:\n{}".format(coupling_weights_sym))
    ode = {
        "x": phases,
        "p": cas.vertcat(
            freqs,
            coupling_weights_sym,
            phases_desired_sym
        ),
        "ode": ode
    }

    integrator = cas.integrator(
        'oscillator_network',
        'cvodes',
        ode,
        {
            "t0": 0,
            "tf": timestep,
            "jit": True,
            # "step0": 1e-3,
            # "abstol": 1e-3,
            # "reltol": 1e-3
        }
    )
    freqs_vals = np.ones(n_dim)
    phases_vals = 1e-1*np.pi*np.random.ranf(n_dim)
    time_tot = 1
    times = np.arange(0, time_tot, timestep)
    phases_logs = np.zeros([len(times), n_dim])
    phases_logs[0, :] = phases_vals
    tic = time.time()
    coupling_weights_vals_reshape = [
        coupling_weights_vals[i, j]
        for i in range(n_dim)
        for j in range(n_dim)
        if isinstance(coupling_weights[i, j], cas.SX)
    ]
    phases_desired_vals_reshape = [
        phases_desired_vals[i, j]
        for i in range(n_dim)
        for j in range(n_dim)
        if isinstance(coupling_weights[i, j], cas.SX)
    ]
    for i, _ in enumerate(times[1:]):
        sim_iteration = i+1
        phases_vals = np.array(
            integrator(
                x0=phases_vals,
                p=np.concatenate([
                    freqs_vals,
                    coupling_weights_vals_reshape,
                    phases_desired_vals_reshape
                ])
            )["xf"][:, 0]
        )
        phases_logs[sim_iteration, :] = phases_vals[:, 0]
    toc = time.time()
    print("Time to simulate {} [s]: {} [s]".format(time_tot, toc-tic))

    plt.figure("Results")
    for i in range(n_dim):
        plt.plot(times, phases_logs[:, i], label="CPG{}".format(i))
        plt.xlabel("Time [s]")
        plt.ylabel("Phase [rad]")
        plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
