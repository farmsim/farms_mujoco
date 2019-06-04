#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import time
import matplotlib.pyplot as plt
from farms_bullet.arenas.arena import FlooredArena, ArenaRamp
from farms_bullet.experiments.salamander.simulation import SalamanderSimulation
from farms_bullet.experiments.salamander.animat_options import SalamanderOptions
from farms_bullet.simulations.simulation_options import SimulationOptions


def run_walking_demos():
    """Run salamander demos"""
    animat_options = SalamanderOptions(
        collect_gps=False,
        show_hydrodynamics=False
    )
    simulation_options = SimulationOptions.with_clargs()
    for arena in [
            FlooredArena(),
            ArenaRamp(ramp_angle=10),
            ArenaRamp(ramp_angle=20),
            ArenaRamp(ramp_angle=40),
            ArenaRamp(ramp_angle=-10),
            ArenaRamp(ramp_angle=-40)
    ]:
        # animat_options.control.drives.forward = 4.9
        sim = SalamanderSimulation(
            simulation_options=simulation_options,
            animat_options=animat_options,
            arena=arena
        )
        sim.run()
        sim.postprocess(
            iteration=sim.iteration,
            plot=simulation_options.plot,
            log_path=simulation_options.log_path,
            log_extension=simulation_options.log_extension,
            record=sim.options.record and not sim.options.headless
        )
        sim.end()


def main():
    """Main"""
    run_walking_demos()
    plt.show()


if __name__ == '__main__':
    TIC = time.time()
    main()
    print("Total simulation time: {} [s]".format(time.time() - TIC))
