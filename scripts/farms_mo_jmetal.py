"""Farms multi-objective optimisation for salamander experiment"""


import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.moead import MOEAD
from jmetal.algorithm.multiobjective.gde3 import GDE3
from jmetal.operator import (
    SBXCrossover,
    PolynomialMutation,
    DifferentialEvolutionCrossover
)
from jmetal.util.observer import (
    ProgressBarObserver,
    VisualizerObserver,
    # BasicObserver
)
from jmetal.util.solutions.evaluator import MultiprocessEvaluator
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.aggregative_function import Tschebycheff
from jmetal.util.solutions import (
    read_solutions,
    print_function_values_to_file,
    print_variables_to_file
)
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.lab.visualization import Plot, InteractivePlot

import matplotlib.pyplot as plt


class SphereFunction(FloatProblem):
    """Benchmark problem"""

    def __init__(self):
        super(SphereFunction, self).__init__()
        self.number_of_objectives = 2
        self.number_of_variables = 2
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = [-5.0 for _ in range(self.number_of_variables)]
        self.upper_bound = [5.0 for _ in range(self.number_of_variables)]

        # FloatSolution.lower_bound = self.lower_bound
        # FloatSolution.upper_bound = self.upper_bound

    @staticmethod
    def get_name():
        """Name"""
        return "Benchmark problem"

    @staticmethod
    def evaluate(solution):
        """Evaluate"""
        np.sum(np.arange(int(1e5)))
        # solution.objectives[0] = 0
        # solution.objectives[1] = 0
        solution.objectives = (
            np.sqrt(
                + (solution.variables[0]-0)**2
                + (solution.variables[1]-1)**2
            ),
            np.sqrt(
                + (solution.variables[0]-1)**2
                + (solution.variables[1]-0)**2
            )
        )
        return solution


def main():
    """Main"""
    n_pop = 100
    n_gen = 5
    problem = SphereFunction()

    max_evaluations = n_pop*n_gen


    # NSGAII
    algorithm = NSGAII(
        problem=problem,
        population_size=n_pop,
        offspring_population_size=n_pop//10,
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


    # # GDE3
    # algorithm = GDE3(
    #     problem=problem,
    #     population_size=n_pop,
    #     cr=0.5,
    #     f=0.5,
    #     termination_criterion=StoppingByEvaluations(max=max_evaluations),
    #     population_evaluator=MultiprocessEvaluator(8),
    # )

    # Visualisers
    algorithm.observable.register(
        observer=ProgressBarObserver(max=max_evaluations)
    )
    algorithm.observable.register(
        observer=VisualizerObserver(
            reference_front=problem.reference_front,
            display_frequency=100
        )
    )
    # algorithm.observable.register(
    #     observer=BasicObserver(frequency=1.0)
    # )

    # Run optimisation
    algorithm.run()

    # Get results
    front = algorithm.get_result()
    ranking = FastNonDominatedRanking()
    pareto_fronts = ranking.compute_ranking(front)

    # Plot front
    plot_front = Plot(
        plot_title='Pareto front approximation',
        reference_front=problem.reference_front,
        axis_labels=problem.obj_labels
    )
    plot_front.plot(
        pareto_fronts[0],
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

    # Optimal
    plt.figure("Decisions")
    plt.plot([1, 0], [0, 1])
    plt.figure("Fitnesses")
    plt.plot([np.sqrt(2), 0], [0, np.sqrt(2)])
    for i, front in enumerate(pareto_fronts):
        # Plot decisions
        plt.figure("Decisions")
        decisions = np.array([solution.variables for solution in front])
        fitness = np.array([solution.objectives for solution in front])
        plt.plot(decisions[:, 0], decisions[:, 1], "o", label=i)
        plt.grid(True)
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        plt.legend()
        # Plot fitnesses
        plt.figure("Fitnesses")
        plt.plot(fitness[:, 0], fitness[:, 1], "o", label=i)
        plt.grid(True)
        plt.legend()
    # Show
    plt.show()


if __name__ == '__main__':
    main()
