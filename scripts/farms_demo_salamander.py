#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import time
from farms_bullet.arenas.arena import FlooredArena, ArenaRamp
from farms_bullet.experiments.salamander.simulation import SalamanderSimulation
from farms_bullet.experiments.salamander.animat_options import SalamanderOptions
from farms_bullet.simulations.simulation_options import SimulationOptions


def run_timestep_demos():
    """Run salamander demos"""
    animat_options = SalamanderOptions(
        collect_gps=False,
        show_hydrodynamics=False
    )
    simulation_options = SimulationOptions.with_clargs()
    for timestep in [0.001, 0.01]:
        simulation_options.timestep = timestep
        simulation_options.video_name = "walking_timestep_{}".format(
            str(timestep).replace(".", "d")
        )
        # animat_options.control.drives.forward = 4.9
        sim = SalamanderSimulation(
            simulation_options=simulation_options,
            animat_options=animat_options
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


def run_walking_demos():
    """Run salamander demos"""
    animat_options = SalamanderOptions(
        collect_gps=False,
        show_hydrodynamics=False
    )
    simulation_options = SimulationOptions.with_clargs()
    for arena_name, arena in [
            ["arena_floor", FlooredArena()],
            ["arena_ramp_p10", ArenaRamp(ramp_angle=10)],
            ["arena_ramp_p20", ArenaRamp(ramp_angle=20)],
            ["arena_ramp_p40", ArenaRamp(ramp_angle=40)],
            ["arena_ramp_n10", ArenaRamp(ramp_angle=-10)],
            ["arena_ramp_n20", ArenaRamp(ramp_angle=-20)],
            ["arena_ramp_n40", ArenaRamp(ramp_angle=-40)]
    ]:
        simulation_options.video_name = arena_name
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


def run_swimming_demos():
    """Run salamander demos"""
    animat_options = SalamanderOptions(
        collect_gps=False,
        show_hydrodynamics=True
    )
    simulation_options = SimulationOptions.with_clargs()
    for drive in [3.001, 3.5, 4, 4.5, 4.999]:
        simulation_options.video_name = "swim_drive_{}".format(
            str(drive).replace(".", "d")
        )
        animat_options.control.drives.forward = drive
        sim = SalamanderSimulation(
            simulation_options=simulation_options,
            animat_options=animat_options
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


def run_gaits_demos():
    """Run salamander demos"""
    animat_options = SalamanderOptions(
        collect_gps=False,
        show_hydrodynamics=False
    )
    simulation_options = SimulationOptions.with_clargs()
    for drive in [0, 1.1, 2, 2.9, 3.1, 4, 4.9, 6]:
        simulation_options.video_name = "gait_drive_{}".format(
            str(drive).replace(".", "d")
        )
        animat_options.control.drives.forward = drive
        sim = SalamanderSimulation(
            simulation_options=simulation_options,
            animat_options=animat_options
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


def run_transition_demo():
    """Run salamander demos"""
    animat_options = SalamanderOptions(
        collect_gps=False,
        show_hydrodynamics=False,
        transition=True
    )
    simulation_options = SimulationOptions.with_clargs()
    simulation_options.duration = 20
    simulation_options.timestep = 0.002
    simulation_options.video_name = "gait_transition"
    sim = SalamanderSimulation(
        simulation_options=simulation_options,
        animat_options=animat_options
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


def run_scale_demos():
    """Run salamander demos"""
    animat_options = SalamanderOptions(
        collect_gps=False,
        show_hydrodynamics=False
    )
    simulation_options = SimulationOptions.with_clargs()
    for scale in [1.0, 0.5, 0.25, 0.1]:
        animat_options.morphology.scale = scale
        simulation_options.video_name = "walking_scale_{}".format(
            str(scale).replace(".", "d")
        )
        # animat_options.control.drives.forward = 4.9
        sim = SalamanderSimulation(
            simulation_options=simulation_options,
            animat_options=animat_options
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
    run_timestep_demos()
    run_walking_demos()
    run_swimming_demos()
    run_gaits_demos()
    run_transition_demo()
    run_scale_demos()


if __name__ == '__main__':
    TIC = time.time()
    main()
    print("Total simulation time: {} [s]".format(time.time() - TIC))
