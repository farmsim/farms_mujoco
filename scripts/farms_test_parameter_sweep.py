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
    return [
        sim.frequency,
        sim.body_stand_amplitude,
        np.linalg.norm(
            sim.experiment_logger.positions.data[-1]
            - sim.experiment_logger.positions.data[0]
        )
    ]


def main():
    """Main"""
    frequencies = np.linspace(0, 3, 10)
    body_amplitudes = np.linspace(0, 0.5, 10)
    parameters = np.array([
        [f, b]
        for f in frequencies
        for b in body_amplitudes
    ])
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
    results_interp = interpolate.interp2d(
        results[:, 0], results[:, 1], results[:, 2],
        kind='linear'  # cubic
    )
    xnew = frequencies
    ynew = body_amplitudes
    xnew_diff2 = 0.5*(xnew[1] - xnew[0])
    ynew_diff2 = 0.5*(ynew[1] - ynew[0])
    znew = results_interp(xnew, ynew)

    extent = (
        min(xnew)-xnew_diff2, max(xnew)+xnew_diff2,
        min(ynew)-ynew_diff2, max(ynew)+ynew_diff2
    )
    plt.plot(parameters[:, 0], parameters[:, 1], "rx")
    plt.imshow(
        znew,
        extent=extent, origin='lower', aspect='auto',
        interpolation="bilinear"
    )
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Body amplitude [rad]")
    cbar = plt.colorbar()
    cbar.set_label('Distance [m]')
    # cbar.set_clim(0, 3)
    plt.show()


if __name__ == '__main__':
    main()
