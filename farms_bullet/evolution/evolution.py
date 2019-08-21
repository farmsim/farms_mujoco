"""Farms multi-objective optimisation"""

# from jmetal.algorithm.multiobjective.moead import MOEAD
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.gde3 import GDE3
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
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.lab.visualization import Plot, InteractivePlot


def run_evolution(problem, n_pop, n_gen):
    """Main"""

    max_evaluations = n_pop*n_gen

    # NSGAII
    algorithm = NSGAII(
        problem=problem,
        population_size=n_pop,
        offspring_population_size=n_pop//2,
        mutation=PolynomialMutation(
            probability=1.0 / problem.number_of_variables,
            distribution_index=0.20  # 20
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
    #     weight_files_path="MOEAD_weights",  # '../../resources/MOEAD_weights',
    #     termination_criterion=StoppingByEvaluations(max=max_evaluations),
    #     population_evaluator=MultiprocessEvaluator(8)
    # )

    # # GDE3
    # algorithm = GDE3(
    #     problem=problem,
    #     population_size=n_pop,
    #     cr=0.7,
    #     f=0.3,
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
            display_frequency=1
        )
    )

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
