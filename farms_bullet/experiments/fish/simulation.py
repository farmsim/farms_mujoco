"""Fish simulation"""

import os

import pybullet

import numpy as np

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
            simulation_options.log_path+"/hydrodynamics.npy",
            sim.elements.animat.data.sensors.hydrodynamics.array
        )

    sim.end()
