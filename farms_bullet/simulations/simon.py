"""Simulation of Simon's experiment"""

import numpy as np

from .simulation import Simulation
from ..experiments.simon import SimonExperiment
from .simulation_options import SimulationOptions
from ..animats.model_options import ModelOptions


class SimonSimulation(Simulation):
    """Simon experiment simulation"""

    def __init__(self, simulation_options, animat_options):
        experiment = SimonExperiment(
            simulation_options,
            len(np.arange(
                0, simulation_options.duration, simulation_options.timestep
            )),
            animat_options=animat_options
        )
        super(SimonSimulation, self).__init__(
            experiment=experiment,
            simulation_options=simulation_options,
            animat_options=animat_options
        )


def run_simon(sim_options=None, animat_options=None):
    """Run Simon's experiment"""

    # Parse command line arguments
    if not sim_options:
        simulation_options = SimulationOptions.with_clargs(duration=100)
    if not animat_options:
        animat_options = ModelOptions()

    # Setup simulation
    print("Creating simulation")
    sim = SimonSimulation(
        simulation_options=simulation_options,
        animat_options=animat_options
    )

    # Run simulation
    print("Running simulation")
    sim.run()

    # Show results
    print("Analysing simulation")
    sim.experiment.postprocess(
        iteration=sim.iteration,
        plot=simulation_options.plot,
        log_path=simulation_options.log_path,
        log_extension=simulation_options.log_extension,
        record=sim.sim_options.record and not sim.sim_options.headless
    )
    sim.end()
