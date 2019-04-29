"""Salamander simulation"""

import numpy as np
from .simulation import Simulation
from .simulation_options import SimulationOptions
from ..animats.model_options import ModelOptions
from ..experiments.salamander import SalamanderExperiment


class SalamanderSimulation(Simulation):
    """Salamander simulation"""

    def __init__(self, simulation_options, animat_options):
        experiment = SalamanderExperiment(
            simulation_options,
            len(np.arange(
                0, simulation_options.duration, simulation_options.timestep
            )),
            animat_options=animat_options
        )
        super(SalamanderSimulation, self).__init__(
            experiment=experiment,
            simulation_options=simulation_options,
            animat_options=animat_options
        )
        self.experiment.save()


def main(simulation_options=None, animat_options=None):
    """Main"""

    # Parse command line arguments
    if not simulation_options:
        simulation_options = SimulationOptions.with_clargs()
    if not animat_options:
        animat_options = ModelOptions()

    # Setup simulation
    print("Creating simulation")
    sim = SalamanderSimulation(
        simulation_options=simulation_options,
        animat_options=animat_options
    )

    # Run simulation
    print("Running simulation")
    sim.run()

    # Analyse results
    print("Analysing simulation")
    sim.experiment.postprocess(
        iteration=sim.iteration,
        plot=simulation_options.plot,
        log_path=simulation_options.log_path,
        log_extension=simulation_options.log_extension,
        record=sim.sim_options.record and not sim.sim_options.headless
    )
    sim.end()


def main_parallel():
    """Simulation with multiprocessing"""
    from multiprocessing import Pool

    # Parse command line arguments
    sim_options = SimulationOptions.with_clargs()

    # Create Pool
    pool = Pool(2)

    # Run simulation
    pool.map(main, [sim_options, sim_options])
    print("Done")


if __name__ == '__main__':
    # main_parallel()
    main()
