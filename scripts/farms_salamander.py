#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import time
import numpy as np
import matplotlib.pyplot as plt
from farms_bullet.experiments.salamander.simulation import main as run_sim
from farms_bullet.animats.amphibious.animat_options import AmphibiousOptions
from farms_bullet.simulations.simulation_options import SimulationOptions


def main():
    """Main"""
    # Animat options
    scale = 1
    animat_options = AmphibiousOptions(
        # collect_gps=True,
        # show_hydrodynamics=True,
        scale=scale
    )
    # animat_options.control.drives.forward = 4

    # Simulation options
    simulation_options = SimulationOptions.with_clargs()
    simulation_options.units.meters = 1
    simulation_options.units.seconds = 1e3
    simulation_options.units.kilograms = 1
    simulation_options.arena = "water"

    # Walking
    animat_options.spawn.position = [0, 0, scale*0.1]
    animat_options.spawn.orientation = [0, 0, 0]
    animat_options.physics.viscous = True
    animat_options.physics.buoyancy = True
    animat_options.physics.water_surface = True
    # Swiming
    # animat_options.spawn.position = [-10, 0, 0]
    # animat_options.spawn.orientation = [0, 0, np.pi]

    # Camera options
    simulation_options.video_yaw = 0
    simulation_options.video_pitch = -30
    simulation_options.video_distance = 1
    # simulation_options.video_name = (
    #     "transition_videos/swim2walk_y{}_p{}_d{}".format(
    #         simulation_options.video_yaw,
    #         simulation_options.video_pitch,
    #         simulation_options.video_distance,
    #     )
    # )

    # Run simulation
    run_sim(
        simulation_options=simulation_options,
        animat_options=animat_options,
        show_progress=True,
    )
    # Show results
    plt.show()


def profile():
    """Profile with cProfile"""
    import cProfile
    import pstats
    cProfile.run("main()", "simulation.profile")
    pstat = pstats.Stats("simulation.profile")
    pstat.sort_stats('time').print_stats(30)
    pstat.sort_stats('cumtime').print_stats(30)


def pycall():
    """Profile with pycallgraph"""
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput
    with PyCallGraph(output=GraphvizOutput()):
        main()


if __name__ == '__main__':
    TIC = time.time()
    # main()
    profile()
    # pycall()
    print("Total simulation time: {} [s]".format(time.time() - TIC))
