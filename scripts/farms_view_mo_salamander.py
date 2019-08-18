"""Farms multiobjective optimisation for salamander"""

import numpy as np
from jmetal.core.solution import FloatSolution
from jmetal.util.ranking import FastNonDominatedRanking
from farms_mo_salamander import SalamanderEvolution


def read_solutions(var_filename, fun_filename, order=0):
    """ Reads a reference front from a file.

    :param filename: File path where the front is located.
    """
    variables = []
    objectives = []
    front = []
    number_of_objectives = 2

    # Variables
    with open(var_filename) as file:
        for line in file:
            variables.append([float(x) for x in line.split()])
    with open(fun_filename) as file:
        for line in file:
            objectives.append([float(x) for x in line.split()])
    variables, objectives = map(np.array, [variables, objectives])
    order = list(np.argsort(objectives[:, 0]))
    print(order)
    for _variables, _objectives in zip(variables[order], objectives[order]):
        solution = FloatSolution([], [], number_of_objectives)
        solution.variables = _variables
        solution.objectives = _objectives
        front.append(solution)
    return front


def run_solutions(solutions, problem_class=SalamanderEvolution):
    """Show simulations """
    for solution in solutions:
        problem = problem_class()
        objectives = solution.objectives.copy()
        print("Objectives: {}".format(objectives))
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
    front = read_solutions(
        "VAR.NSGAII.Salamander evolution",
        "FUN.NSGAII.Salamander evolution"
        # "VAR.GDE3.Salamander evolution",
        # "FUN.GDE3.Salamander evolution"
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
