"""Polypterus simulation"""

from ...animats.amphibious.simulation import AmphibiousSimulation
from .animat import Polypterus


class PolypterusSimulation(AmphibiousSimulation):
    """Polypterus simulation"""

    def __init__(self, simulation_options, animat_options, *args, **kwargs):
        animat = Polypterus(
            animat_options,
            simulation_options.timestep,
            simulation_options.n_iterations,
            simulation_options.units
        )
        super(PolypterusSimulation, self).__init__(
            simulation_options,
            animat,
            *args,
            **kwargs
        )


def main(simulation_options=None, animat_options=None):
    """Main"""

    # Parse command line arguments
    if not simulation_options:
        simulation_options = SimulationOptions.with_clargs()
    if not animat_options:
        animat_options = AmphibiousOptions()

    # Setup simulation
    print("Creating simulation")
    sim = PolypterusSimulation(
        simulation_options=simulation_options,
        animat_options=animat_options
    )

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


if __name__ == '__main__':
    # main_parallel()
    main()
