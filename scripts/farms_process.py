#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import time
from farms_bullet.arenas.arena import FlooredArena, ArenaRamp
from farms_bullet.experiments.salamander.simulation import SalamanderSimulation
from farms_bullet.experiments.salamander.animat_options import SalamanderOptions
from farms_bullet.simulations.simulation_options import SimulationOptions


def run_experiment():
    """Run salamander demos"""
    animat_options = SalamanderOptions(
        collect_gps=True,
        show_hydrodynamics=False
    )
    simulation_options = SimulationOptions.with_clargs()
    animat_options.morphology.scale = 1
    animat_options.control.drives.forward = 2
    animat_options.control.drives.left = 0
    animat_options.control.drives.right = 0
    simulation_options.duration = 20
    simulation_options.timestep = 0.001
    #arena = ArenaRamp(ramp_angle=0)
    arena = FlooredArena()
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
    run_experiment()


if __name__ == '__main__':
    TIC = time.time()
    main()
    print("Total simulation time: {} [s]".format(time.time() - TIC))
