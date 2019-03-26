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
        sim.body_amplitude,
        np.linalg.norm(
            sim.experiment_logger.positions.data[:-1]
            - sim.experiment_logger.positions.data[0]
        )
    ]


def main():
    """Main"""
    frequencies = np.linspace(0, 3, 10)
    body_amplitudes = np.linspace(0, 0.3, 5)
    parameters = np.array([
        [f, b]
        for f in frequencies
        for b in body_amplitudes
    ])
    simulations_options = [
        SimulationOptions.with_clargs(
            headless=True,
            fast=True,
            duration=1,
            frequency=frequency,
            body_amplitude=body_amplitude
        )
        for frequency in frequencies
        for body_amplitude in body_amplitudes
    ]
    pool = Pool(4)
    results = np.array(pool.map(run_simulation, simulations_options))
    print("DONE: Completed parameter sweep")
    print(results)
    results_interp = interpolate.interp2d(
        results[:, 0], results[:, 1], results[:, 2],
        kind='linear'  # cubic
    )
    xnew = np.linspace(min(frequencies), max(frequencies), 100)
    ynew = np.linspace(min(body_amplitudes), max(body_amplitudes), 100)
    znew = results_interp(xnew, ynew)

    extent = (min(xnew), max(xnew), min(ynew), max(ynew))
    plt.plot(parameters[:, 0], parameters[:, 1], "rx")
    plt.imshow(znew, extent=extent, origin='lower', aspect='auto')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Body amplitude [rad]")
    cbar = plt.colorbar()
    cbar.set_label('Distance [m]')
    plt.show()


if __name__ == '__main__':
    main()
