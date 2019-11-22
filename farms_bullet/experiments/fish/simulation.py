"""Fish simulation"""

import pybullet

import numpy as np

from ...animats.amphibious.simulation import AmphibiousSimulation
from .animat import Fish


class FishSimulation(AmphibiousSimulation):
    """Fish simulation"""

    def __init__(self, fish_name, fish_version, simulation_options, animat_options, *args, **kwargs):
        animat = Fish.from_fish_data(
            fish_name,
            fish_version,
            animat_options,
            simulation_options.timestep,
            simulation_options.n_iterations,
            simulation_options.units
        )
        super(FishSimulation, self).__init__(
            simulation_options,
            animat,
            *args,
            **kwargs
        )


def main(fish_name, fish_version, simulation_options, animat_options):
    """Main"""

    # Setup simulation
    print("Creating simulation")
    sim = FishSimulation(
        fish_name,
        fish_version,
        simulation_options=simulation_options,
        animat_options=animat_options
    )
    pybullet.setGravity(0, 0, 0)

    # Run simulation
    print("Running simulation")
    sim.run()

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
