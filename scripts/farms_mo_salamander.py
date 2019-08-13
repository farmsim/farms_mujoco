"""Farms multiobjective optimisation for salamander"""

import numpy as np
from jmetal.core.problem import FloatProblem
# from jmetal.algorithm.multiobjective.moead import MOEAD
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import (
    SBXCrossover,
    PolynomialMutation,
    DifferentialEvolutionCrossover
)
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solutions.evaluator import MultiprocessEvaluator
from jmetal.util.termination_criterion import StoppingByEvaluations
# from jmetal.util.aggregative_function import Tschebycheff
from jmetal.util.solutions import (
    # read_solutions,
    print_function_values_to_file,
    print_variables_to_file
)
from jmetal.lab.visualization import Plot, InteractivePlot

from farms_bullet.simulations.simulation_options import SimulationOptions
from farms_bullet.experiments.salamander.simulation import SalamanderSimulation
from farms_bullet.experiments.salamander.animat_options import SalamanderOptions


class SalamanderEvolution(FloatProblem):
    """Benchmark problem"""

    def __init__(self):
        super(SalamanderEvolution, self).__init__()
        self.number_of_objectives = 2
        self.number_of_variables = 2
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["Distance (negative)", "Consumption"]

        self.lower_bound = [-np.inf for _ in range(self.number_of_variables)]
        self.upper_bound = [+np.inf for _ in range(self.number_of_variables)]

        # Body stand amplitude
        self.lower_bound[0], self.upper_bound[0] = 0, 2*np.pi/11

    @staticmethod
    def get_name():
        """Name"""
        return "Salamander evolution"

    @staticmethod
    def evaluate(solution):
        """Evaluate"""
        # Animat options
        animat_options = SalamanderOptions(
            # collect_gps=True,
            scale=1
        )
        animat_options.control.network.oscillators.body_head_amplitude = 0
        animat_options.control.network.oscillators.body_tail_amplitude = 0
        animat_options.control.network.oscillators.set_body_stand_amplitude(
            solution.variables[0]
        )
        animat_options.control.network.oscillators.body_stand_shift = np.pi/4
        # animat_options.control.drives.forward = 4
        # Simulation options
        simulation_options = SimulationOptions.with_clargs()
        simulation_options.headless = True
        simulation_options.fast = True
        simulation_options.timestep = 1e-2
        simulation_options.duration = 10
        simulation_options.units.meters = 1
        simulation_options.units.seconds = 1
        simulation_options.units.kilograms = 1
        simulation_options.plot = False

        # Setup simulation
        sim = SalamanderSimulation(
            simulation_options=simulation_options,
            animat_options=animat_options
        )

        # Run simulation
        sim.run()

        # Extract fitness
        total_torque = np.sum(np.abs(
            sim.elements.animat.data.sensors.proprioception.motor_torques()
        ))*simulation_options.timestep/simulation_options.duration
        position = sim.elements.animat.data.sensors.gps.urdf_position(
            iteration=sim.iteration-1,
            link_i=0
        )
        solution.objectives[0] = position[0]  # Distance along x axis
        solution.objectives[1] = total_torque  # Energy

        # Terminate simulation
        sim.end()

        return solution


def main():
    """Main"""
    n_pop = 20
    n_gen = 3
    problem = SalamanderEvolution()

    max_evaluations = n_pop*n_gen

    # NSGAII
    algorithm = NSGAII(
        problem=problem,
        population_size=n_pop,
        offspring_population_size=n_pop//2,
        mutation=PolynomialMutation(
            probability=1.0 / problem.number_of_variables,
            distribution_index=20
        ),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max=max_evaluations),
        population_evaluator=MultiprocessEvaluator(8)
    )

    # # MOEAD
    # algorithm = MOEAD(
    #     problem=problem,
    #     population_size=n_pop,
    #     crossover=DifferentialEvolutionCrossover(CR=1.0, F=0.5, K=0.5),
    #     mutation=PolynomialMutation(
    #         probability=1.0 / problem.number_of_variables,
    #         distribution_index=20
    #     ),
    #     aggregative_function=Tschebycheff(
    #         dimension=problem.number_of_objectives
    #     ),
    #     neighbor_size=n_pop//5,
    #     neighbourhood_selection_probability=0.9,
    #     max_number_of_replaced_solutions=2,
    #     weight_files_path='../../resources/MOEAD_weights',
    #     termination_criterion=StoppingByEvaluations(max=max_evaluations),
    #     population_evaluator=MultiprocessEvaluator(8)
    # )

    # Visualisers
    algorithm.observable.register(
        observer=ProgressBarObserver(max=max_evaluations)
    )
    algorithm.observable.register(
        observer=VisualizerObserver(
            reference_front=problem.reference_front,
            display_frequency=1
        )
    )

    # Run optimisation
    algorithm.run()

    # Get results
    front = algorithm.get_result()

    # Plot front
    plot_front = Plot(
        plot_title='Pareto front approximation',
        reference_front=problem.reference_front,
        axis_labels=problem.obj_labels
    )
    plot_front.plot(
        front,
        label=algorithm.label,
        filename=algorithm.get_name()
    )

    # Plot interactive front
    plot_front = InteractivePlot(
        plot_title='Pareto front approximation interactive',
        reference_front=problem.reference_front,
        axis_labels=problem.obj_labels
    )
    plot_front.plot(
        front,
        label=algorithm.label,
        filename=algorithm.get_name()
    )

    # Save results to file
    print_function_values_to_file(front, 'FUN.' + algorithm.label)
    print_variables_to_file(front, 'VAR.' + algorithm.label)

    # Print information
    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))


if __name__ == '__main__':
    main()
