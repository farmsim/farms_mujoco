#!/usr/bin/env python3
"""Run crescent_gunnel simulation with bullet"""

import os
import time
import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt
from farms_bullet.experiments.crescent_gunnel.simulation import main as run_sim
from farms_bullet.animats.amphibious.animat_options import AmphibiousOptions
from farms_bullet.simulations.simulation_options import SimulationOptions


def main():
    """Main"""
    # Animat options
    scale = 1
    animat_options = AmphibiousOptions(
        # collect_gps=True,
        show_hydrodynamics=True,
        scale=scale,
        n_joints_body=20,
        viscous=True,
        viscous_coefficients=[
            1e-1*np.array([-1e-4, -5e-1, -3e-1]),
            1e-1*np.array([-1e-6, -1e-6, -1e-6])
        ],
        water_surface=False
    )
    # animat_options.control.drives.forward = 4

    # Simulation options
    simulation_options = SimulationOptions.with_clargs()
    simulation_options.units.meters = 1
    simulation_options.units.seconds = 1e3
    simulation_options.units.kilograms = 1

    # Kinematics
    animat_options.control.kinematics_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "farms_bullet",
        "experiments",
        "crescent_gunnel",
        "kinematics",
        "kinematics.csv"
    )
    kinematics = np.loadtxt(animat_options.control.kinematics_file)
    pose = kinematics[:, :3]
    n_samples = 10*np.shape(kinematics)[0]
    pose *= 1e-3
    pose = resample(pose, n_samples)
    position = np.ones(3)
    position[:2] = pose[0, :2]
    orientation = np.zeros(3)
    orientation[2] = pose[0, 2] + np.pi
    velocity = np.zeros(3)
    n_sample = 100
    velocity[:2] = pose[n_sample, :2] - pose[0, :2]
    velocity /= n_sample*simulation_options.timestep
    kinematics = kinematics[:, 3:]
    kinematics = ((kinematics + np.pi) % (2*np.pi)) - np.pi
    kinematics = resample(kinematics, n_samples)

    # Walking
    animat_options.spawn.position = position
    animat_options.spawn.orientation = orientation
    animat_options.physics.buoyancy = False
    animat_options.spawn.velocity_lin = velocity
    animat_options.spawn.velocity_ang = [0, 0, 0]
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
        animat_options=animat_options
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
