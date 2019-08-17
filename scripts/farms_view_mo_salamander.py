"""Farms multiobjective optimisation for salamander"""

import numpy as np

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
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
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.lab.visualization import Plot, InteractivePlot

from farms_bullet.simulations.simulation_options import SimulationOptions
from farms_bullet.experiments.salamander.simulation import SalamanderSimulation
from farms_bullet.experiments.salamander.animat_options import SalamanderOptions

from farms_mo_salamander import SalamanderEvolution


def read_solutions(var_filename, fun_filename):
    """ Reads a reference front from a file.

    :param filename: File path where the front is located.
    """
    front = []
    number_of_objectives = 2

    with open(var_filename) as file:
        for line in file:
            vector = [float(x) for x in line.split()]
            solution = FloatSolution([], [], number_of_objectives)
            solution.variables = vector
            front.append(solution)
    with open(fun_filename) as file:
        for i, line in enumerate(file):
            vector = [float(x) for x in line.split()]
            front[i].objectives = vector
    return front


def run_solutions(solutions, problem=SalamanderEvolution):
    """Show simulations """
    for solution in solutions:
        objectives = solution.objectives.copy()
        solution2 = problem.evaluate(solution, evolution=False)
        same_objectives = all([
            obj1 == obj2
            for obj1, obj2
            in zip(objectives, solution2.objectives)
        ])
        print("Verifying objectives:")
        print("  Previous objective: {}".format(objectives))
        print("  New objective: {}".format(solution2.objectives))
        print("  Same objectives: {}".format(same_objectives))
        solution.objectives = objectives.copy()


def main():
    """Main"""

    # Load results from file
    # print_function_values_to_file(front, 'FUN.' + algorithm.label)
    # print_variables_to_file(front, 'VAR.' + algorithm.label)
    front = read_solutions(
        "VAR.NSGAII.Salamander evolution",
        "FUN.NSGAII.Salamander evolution"
    )
    ranking = FastNonDominatedRanking()
    pareto_fronts = ranking.compute_ranking(front)

    # Visualise best results (Pareto front)
    print("Found {} interesting solutions in pareto_front".format(
        len(pareto_fronts[0])
    ))

    run_solutions(pareto_fronts[0].copy())

    # Print solutions information
    print("Solutions:")
    for i, sol in enumerate(front):
        print("  Solution {}".format(i))
        print("    Variables: {}".format(sol.variables))
        print("    Objectives: {}".format(sol.objectives))
    print("Pareto front:")
    for i, sol in enumerate(pareto_fronts[0]):
        print("  Solution {}".format(i))
        print("    Variables: {}".format(sol.variables))
        print("    Objectives: {}".format(sol.objectives))


if __name__ == '__main__':
    main()
