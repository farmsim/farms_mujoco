"""Farms - Test salamander network"""

import time
import numpy as np
import matplotlib.pyplot as plt
from farms_bullet.controllers.network import SalamanderNetworkODE


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
    # Allocation
    tic = time.time()
    timestep = 1e-3
    times = np.arange(0, 10, timestep)
    network = SalamanderNetworkODE.walking(
        n_iterations=len(times),
        timestep=timestep
    )
    n_iterations = len(times)
    freqs = 2*np.pi*np.ones(np.shape(network.phases)[1])
    toc = time.time()
    print("Time to allocate data: {} [s]".format(toc-tic))

    # Simulate (method 1)
    time_control = 0
    for _ in range(n_iterations-1):
        tic0 = time.time()
        network.control_step(freqs)
        tic1 = time.time()
        time_control += tic1 - tic0
    print("Integration time: {} [s]".format(time_control))

    # Plot phase
    plot_data(
        times,
        network.phases,
        22,
        "Phases",
        r"$\theta{}$",
        "Phase [rad]"
    )

    # Plot amplitude
    plot_data(
        times,
        network.amplitudes,
        22,
        "Amplitudes",
        r"$r{}$",
        "Amplitude [rad]"
    )

    # Plot dphase
    plot_data(
        times,
        network.dphases,
        22,
        "dPhases",
        r"$d\theta{}$",
        "dPhase [rad]"
    )

    # Plot damplitude
    plot_data(
        times,
        network.damplitudes,
        22,
        "dAmplitudes",
        r"$dr{}$",
        "dAmplitude [rad]"
    )

    # Plot output
    plot_data(
        times,
        network.get_outputs_all(),
        22,
        "Outputs",
        r"$r{}$",
        "Output"
    )

    # Plot doutput
    plot_data(
        times,
        network.get_doutputs_all(),
        22,
        "dOutputs",
        r"$r{}$",
        "dOutput"
    )

    # Plot positions
    plot_data(
        times,
        network.get_position_output_all(),
        11,
        "Positions",
        r"$\theta{}$",
        "Position [rad]"
    )

    # Plot velocitys
    plot_data(
        times,
        network.get_velocity_output_all(),
        11,
        "Velocitys",
        r"$\theta{}$",
        "Velocity [rad]"
    )

    plt.show()


if __name__ == '__main__':
    # import pdb
    # pdb.run("main()")
    main()
