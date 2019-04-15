"""Farms - Test salamander network"""

import time
import numpy as np
import matplotlib.pyplot as plt
from farms_bullet.controllers.network import SalamanderNetworkPosition


def plot_data(times, data, body_ids, figurename, label, ylabel):
    """Plot data"""
    for i, _data in enumerate(data.T):
        if i < body_ids:
            plt.figure("{}-body".format(figurename))
        else:
            plt.figure("{}-legs".format(figurename))
        plt.plot(times, _data, label=r"{}{}".format(label, i))
    for figure in ["{}-body", "{}-legs"]:
        plt.figure(figure.format(figurename))
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel(ylabel)
        plt.legend()


def main():
    """Main"""
    timestep = 1e-3
    times = np.arange(0, 10, timestep)
    n_iterations = len(times)
    network = SalamanderNetworkPosition.pos_walking(timestep=1e-3)
    freqs = np.ones(np.shape(network.phases))
    phase_log = np.zeros([n_iterations, len(network.phases)])
    dphase_log = np.zeros([n_iterations, len(network.dphases)])
    amplitude_log = np.zeros([n_iterations, len(network.amplitudes)])
    damplitude_log = np.zeros([n_iterations, len(network.damplitudes)])
    output_log = np.zeros([n_iterations, len(network.get_outputs())])
    position_log = np.zeros([n_iterations, len(network.get_position_output())])
    velocity_log = np.zeros([n_iterations, len(network.get_velocity_output())])

    # Simulate
    time_control = 0
    time_log = 0
    time_log2 = 0
    for i in range(n_iterations):
        tic0 = time.time()
        network.control_step(freqs)
        tic1 = time.time()
        phase_log[i, :] = network.phases
        amplitude_log[i, :] = network.amplitudes
        dphase_log[i, :] = network.dphases
        damplitude_log[i, :] = network.damplitudes
        output_log[i, :] = network.get_outputs()
        position_log[i, :] = network.get_position_output()
        tic2 = time.time()
        velocity_log[i, :] = network.get_velocity_output()
        tic3 = time.time()
        time_control += tic1 - tic0
        time_log += tic2 - tic1
        time_log2 += tic3 - tic2
    print("Integration time: {} [s]".format(time_control))
    print("Log time: {} [s]".format(time_log))
    print("Log2 time: {} [s]".format(time_log2))

    # Plot phase
    plot_data(
        times,
        phase_log,
        22,
        "Phases",
        r"$\theta{}$",
        "Phase [rad]"
    )

    # Plot amplitude
    plot_data(
        times,
        amplitude_log,
        22,
        "Amplitudes",
        r"$r{}$",
        "Amplitude [rad]"
    )

    # Plot dphase
    plot_data(
        times,
        dphase_log,
        22,
        "dPhases",
        r"$d\theta{}$",
        "dPhase [rad]"
    )

    # Plot damplitude
    plot_data(
        times,
        damplitude_log,
        22,
        "dAmplitudes",
        r"$dr{}$",
        "dAmplitude [rad]"
    )

    # Plot output
    plot_data(
        times,
        output_log,
        22,
        "Outputs",
        r"$r{}$",
        "Output"
    )

    # Plot positions
    plot_data(
        times,
        position_log,
        11,
        "Positions",
        r"$\theta{}$",
        "Position [rad]"
    )

    # Plot velocitys
    plot_data(
        times,
        velocity_log,
        11,
        "Velocitys",
        r"$\theta{}$",
        "Velocity [rad]"
    )

    plt.show()


if __name__ == '__main__':
    main()
