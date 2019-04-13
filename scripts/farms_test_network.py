"""Farms - Test salamander network"""

import time
import numpy as np
import matplotlib.pyplot as plt
from farms_bullet.controllers.network import SalamanderNetworkPosition


def main():
    """Main"""
    timestep = 1e-3
    times = np.arange(0, 10, timestep)
    n_iterations = len(times)
    network = SalamanderNetworkPosition.from_gait(gait="walking", timestep=1e-3)
    freqs = np.ones(np.shape(network.phases))
    tic = time.time()
    phase_log = np.zeros([n_iterations, len(network.phases)])
    amplitude_log = np.zeros([n_iterations, len(network.amplitudes)])
    output_log = np.zeros([n_iterations, len(network.outputs)])
    position_log = np.zeros([n_iterations, len(network.get_position_output())])

    # Simulate
    for i in range(n_iterations):
        network.control_step(freqs)
        phase_log[i, :] = network.phases
        amplitude_log[i, :] = network.amplitudes
        output_log[i, :] = network.outputs
        position_log[i, :] = network.get_position_output()
    print("Integration time: {} [s]".format(time.time() - tic))

    # Plot phases
    for i, data in enumerate(phase_log.T):
        if i < 22:
            plt.figure("Phases-body")
        else:
            plt.figure("Phases-legs")
        plt.plot(times, data, label=r"$\theta{}$".format(i))
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Phase [rad]")
        plt.legend()

    # Plot ampltiude
    for i, data in enumerate(amplitude_log.T):
        if i < 22:
            plt.figure("Amplitudes-body")
        else:
            plt.figure("Amplitudes-legs")
        plt.plot(times, data, label=r"$r{}$".format(i))
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude [rad]")
        plt.legend()

    # Plot ampltiude
    for i, data in enumerate(output_log.T):
        if i < 22:
            plt.figure("Outputs-body")
        else:
            plt.figure("Outputs-legs")
        plt.plot(times, data, label=r"$r{}$".format(i))
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Output")
        plt.legend()

    # Plot positions
    for i, data in enumerate(position_log.T):
        if i < 11:
            plt.figure("Positions-body")
        else:
            plt.figure("Positions-legs")
        plt.plot(times, data, label=r"$\theta{}$".format(i))
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Position [rad]")
        plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
