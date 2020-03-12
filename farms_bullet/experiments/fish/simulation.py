"""Fish simulation"""

import os

import pybullet

import numpy as np
from scipy.interpolate import interp1d

from ...animats.amphibious.animat_options import AmphibiousOptions
from ...simulations.simulation_options import SimulationOptions
from ...animats.amphibious.simulation import AmphibiousSimulation
from .animat import Fish


class FishSimulation(AmphibiousSimulation):
    """Fish simulation"""

    def __init__(self, sdf_path, simulation_options, animat_options, *args, **kwargs):
        animat = Fish(
            animat_options,
            simulation_options.timestep,
            simulation_options.n_iterations,
            simulation_options.units,
            sdf_path=sdf_path
        )
        super(FishSimulation, self).__init__(
            simulation_options,
            animat,
            *args,
            **kwargs
        )


FISH_DIRECTORY = os.path.dirname(os.path.realpath(__file__))


def main(sdf_path, simulation_options, animat_options, show_progress=False):
    """Main"""

    # Setup simulation
    print("Creating simulation")
    sim = FishSimulation(
        sdf_path,
        simulation_options=simulation_options,
        animat_options=animat_options
    )
    pybullet.setGravity(0, 0, 0)

    # Run simulation
    print("Running simulation")
    sim.run(show_progress=show_progress)

    # Analyse results
    print("Analysing simulation")
    sim.postprocess(
        iteration=sim.iteration,
        plot=simulation_options.plot,
        log_path=simulation_options.log_path,
        log_extension=simulation_options.log_extension,
        record=sim.options.record and not sim.options.headless
    )
    if simulation_options.log_path:
        np.save(
            os.path.join(simulation_options.log_path, 'hydrodynamics.npy'),
            sim.elements.animat.data.sensors.hydrodynamics.array
        )

    sim.end()
    return sim


def iterator(sdf_path, simulation_options, animat_options, show_progress=False):
    """Main"""

    # Setup simulation
    print("Creating simulation")
    sim = FishSimulation(
        sdf_path,
        simulation_options=simulation_options,
        animat_options=animat_options
    )
    pybullet.setGravity(0, 0, 0)
    return sim


def fish_simulation(kinematics_file, sdf_path, results_path, **kwargs):
    """Fish simulation"""
    # Animat options
    scale = 1
    animat_options = AmphibiousOptions(
        # collect_gps=True,
        show_hydrodynamics=True,
        scale=scale,
        n_joints_body=20,
        viscous=False,
        resistive=True,
        resistive_coefficients=kwargs.pop(
            'resistive_coefficients',
            [
                1e-1*np.array([-1e-4, -5e-1, -3e-1]),
                1e-1*np.array([-1e-6, -1e-6, -1e-6])
            ]
        ),
        water_surface=False
    )
    # animat_options.control.drives.forward = 4

    # Simulation options
    simulation_options = SimulationOptions.with_clargs()
    simulation_options.units.meters = 1
    simulation_options.units.seconds = 1e3
    simulation_options.units.kilograms = 1
    simulation_options.fast = kwargs.pop('fast', False)
    simulation_options.headless = kwargs.pop('headless', False)

    # Kinematics
    animat_options.control.kinematics_file = kinematics_file
    original_kinematics = np.loadtxt(animat_options.control.kinematics_file)
    len_kinematics = np.shape(original_kinematics)[0]
    simulation_options.duration = (len_kinematics-1)*1e-2
    pose = original_kinematics[:, :3]
    # pose *= 1e-3
    # pose *= 1e-3
    # pose[0, :2] *= 1e-3
    # pose[0, 2] *= 1e-3
    position = np.ones(3)
    position[:2] = pose[0, :2]
    orientation = kwargs.pop('orientation', None)
    if orientation is None:
        orientation = np.zeros(3)
        orientation[2] = pose[0, 2]
    velocity = kwargs.pop('velocity', None)
    if velocity is None:
        velocity = np.zeros(3)
        n_sample = 5 if pose.shape[0] > 3 else (pose.shape[0]-1)
        velocity[:2] = pose[n_sample, :2] - pose[0, :2]
        sampling_timestep = 1e-2
        velocity /= n_sample*sampling_timestep
    kinematics = np.copy(original_kinematics)
    kinematics[:, 3:] = ((kinematics[:, 3:] + np.pi) % (2*np.pi)) - np.pi
    n_iterations = (len_kinematics-1)*10+1
    interp_x = np.arange(0, n_iterations, 10)
    interp_xn = np.arange(n_iterations)
    kinematics = interp1d(
        interp_x,
        kinematics,
        axis=0
    )(interp_xn)

    # Swimming
    animat_options.spawn.position = position
    animat_options.spawn.orientation = orientation
    animat_options.physics.buoyancy = False
    animat_options.spawn.velocity_lin = velocity
    animat_options.spawn.velocity_ang = [0, 0, 0]
    animat_options.spawn.joints_positions = kinematics[0, 3:]

    # Logging
    simulation_options.log_path = results_path

    # Camera options
    simulation_options.video_yaw = 0
    simulation_options.video_pitch = -30
    simulation_options.video_distance = 1
    # simulation_options.video_name = (
    #     'transition_videos/swim2walk_y{}_p{}_d{}'.format(
    #         simulation_options.video_yaw,
    #         simulation_options.video_pitch,
    #         simulation_options.video_distance,
    #     )
    # )


    # Run simulation
    if kwargs.pop('iterator', False):
        sim = iterator(
            sdf_path,
            simulation_options=simulation_options,
            animat_options=animat_options,
            show_progress=True
        )
    else:
        sim = main(
            sdf_path,
            simulation_options=simulation_options,
            animat_options=animat_options,
            show_progress=True
        )
    assert not kwargs, kwargs
    return sim, kinematics
