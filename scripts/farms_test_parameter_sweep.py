#!/usr/bin/env python3
"""Test parameter sweeps using salamander simulation with bullet"""

from multiprocessing import Pool
from farms_bullet.simulation import Simulation
from farms_bullet.simulation_options import SimulationOptions
import numpy as np


def run_simulation(options):
    """Run simulation"""
    sim = Simulation(options=options)
    sim.run(options)


def main():
    """Main"""
    simulations_options = [
        SimulationOptions.with_clargs(
            headless=True,
            fast=True,
            duration=1,
            frequency=frequency,
            body_amplitude=body_amplitude
        )
        for frequency in np.linspace(0.5, 2, 8)
        for body_amplitude in np.linspace(0.1, 0.3, 5)
    ]
    pool = Pool(4)
    pool.map(run_simulation, simulations_options)
    print("DONE: Completed parameter sweep")


if __name__ == '__main__':
    main()
