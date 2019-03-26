#!/usr/bin/env python3
"""Test parameter sweeps using salamander simulation with bullet"""

from multiprocessing import Pool
from farms_bullet.simulation import Simulation
from farms_bullet.simulation_options import SimulationOptions
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


def run_simulation(options):
    """Run simulation"""
    sim = Simulation(options=options)
    sim.run(options)
    distance_traveled = np.linalg.norm(
        sim.experiment_logger.positions.data[-1]
        - sim.experiment_logger.positions.data[0]
    )
    torques_sum = (np.abs(sim.experiment_logger.motors.joints_cmds())).sum()
    print("Distance traveled: {} [m]".format(distance_traveled))
    print("Torques sum: {} [Nm]".format(torques_sum))
    return [
        sim.frequency,
        sim.body_stand_amplitude,
        distance_traveled,
        torques_sum,
        distance_traveled/torques_sum,
        distance_traveled/torques_sum**2,
        distance_traveled**2/torques_sum
    ]


def plot_result(results, index, xnew, ynew, figure_name, label):
    """Plot result"""
    plt.figure(figure_name)
    results_interp = interpolate.interp2d(
        results[:, 0], results[:, 1], results[:, index],
        kind='linear'  # cubic
    )
    xnew_diff2 = 0.5*(xnew[1] - xnew[0])
    ynew_diff2 = 0.5*(ynew[1] - ynew[0])
    znew = results_interp(xnew, ynew)

    extent = (
        min(xnew)-xnew_diff2, max(xnew)+xnew_diff2,
        min(ynew)-ynew_diff2, max(ynew)+ynew_diff2
    )
    parameters = np.array([
        [x, y]
        for x in xnew
        for y in ynew
    ])
    plt.plot(parameters[:, 0], parameters[:, 1], "rx")
    plt.imshow(
        znew,
        extent=extent, origin='lower', aspect='auto',
        interpolation="bilinear"
    )
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Body amplitude [rad]")
    cbar = plt.colorbar()
    cbar.set_label(label)
    # cbar.set_clim(0, 3)
    


def main():
    """Main"""
    frequencies = np.linspace(0, 3, 10)
    body_amplitudes = np.linspace(0, 0.5, 10)
    simulations_options = [
        SimulationOptions.with_clargs(
            headless=True,
            fast=True,
            duration=2,
            frequency=frequency,
            body_stand_amplitude=body_stand_amplitude
        )
        for frequency in frequencies
        for body_stand_amplitude in body_amplitudes
    ]
    pool = Pool(4)
    results = np.array(pool.map(run_simulation, simulations_options))
    print("DONE: Completed parameter sweep")
    print(results)
    plot_result(
        results, index=2,
        xnew=frequencies, ynew=body_amplitudes,
        figure_name="Distance", label='Distance [m]'
    )
    plot_result(
        results, index=3,
        xnew=frequencies, ynew=body_amplitudes,
        figure_name="Torques", label='Torque sum [Nm]'
    )
    plot_result(
        results, index=4,
        xnew=frequencies, ynew=body_amplitudes,
        figure_name="d/t", label='Dist/torques'
    )
    plot_result(
        results, index=5,
        xnew=frequencies, ynew=body_amplitudes,
        figure_name="d/t2", label='Dist/torques**2'
    )
    plot_result(
        results, index=6,
        xnew=frequencies, ynew=body_amplitudes,
        figure_name="d2/t", label='Dist**2/torques'
    )
    plt.show()


if __name__ == '__main__':
    main()
