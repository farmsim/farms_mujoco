"""Farms multi-objective optimisation for salamander experiment"""

import numpy as np
import platypus as pla
import matplotlib.pyplot as plt

from farms_bullet.evolution.evolution import Evolution, ProblemLogger


class SalamanderEvolution(pla.Problem):
    """Salamander evolution"""

    def __init__(self, n_evaluations):
        n_vars, n_objs = 2, 2
        super(SalamanderEvolution, self).__init__(nvars=n_vars, nobjs=n_objs)
        self.types[0] = pla.Real(-10, 10)
        self.types[1] = pla.Real(-10, 10)
        self.logger = ProblemLogger(n_evaluations, n_vars, n_objs)

    def evaluate(self, solution):
        solution.objectives[0] = np.sqrt(
            + (solution.variables[0]-0)**2
            + (solution.variables[1]-1)**2
        )
        solution.objectives[1] = np.sqrt(
            + (solution.variables[0]-1)**2
            + (solution.variables[1]-0)**2
        )
        self.logger.log(solution.variables, solution.objectives)


def main():
    """Main"""
    n_population = 10  # int(1e2)
    n_generations = 1  # int(1e2)
    n_runs = 10
    algorithm = pla.MOEAD
    evolution = Evolution(
        problem=SalamanderEvolution,
        algorithms=[
            algorithm,
            algorithm,
            algorithm,
            algorithm

            # pla.GDE3,
            # pla.GDE3,
            # pla.GDE3,
            # pla.GDE3,qq

            # pla.NSGAIII,
            # pla.NSGAIII,
            # pla.NSGAIII,
            # pla.NSGAIII,

            # pla.NSGAII,
            # pla.NSGAII,
            # pla.NSGAII,
            # pla.NSGAII,

            # pla.SMPSO,
            # pla.SMPSO,
            # pla.SMPSO,
            # pla.SMPSO,

            # pla.CMAES,
            # pla.CMAES,
            # pla.CMAES,
            # pla.CMAES,

            # pla.NSGAII,
            # pla.CMAES,
            # pla.GDE3,
            # pla.SMPSO,

            # Others:
            # pla.NSGAII,
            # (pla.NSGAIII, {"divisions_outer":12}),
            # (pla.CMAES, {"epsilons":[0.05]}),
            # pla.GDE3,
            # pla.IBEA,
            # (pla.MOEAD, {
            #     # "weight_generator": normal_boundary_weights,
            #     "divisions_outer":12
            # }),
            # (pla.OMOPSO, {"epsilons":[0.05]}),
            # pla.SMPSO,
            # pla.SPEA2,
            # (pla.EpsMOEA, {"epsilons":[0.05]})
        ],
        n_population=n_population,
        n_generations=n_generations,
        n_runs=n_runs,
        options={
            # # NSGA
            # "selector": pla.TournamentSelector(
            #     tournament_size=30,
            #     dominance=pla.ParetoDominance()
            # ),
            # "variator": pla.GAOperator(
            #     pla.SBX(0., 10.0),
            #     pla.PM(0.5, 100.0)
            # ),
            # DE
            "variator": pla.DifferentialEvolution(
                crossover_rate=0.1,
                step_size=0.5
            ),
            # "divisions_outer": 100
            # "neighborhood_size": 10
        }
    )
    evolution.run_evolution()
    print("Evolution complete")
    print("Number of evaluations: {}".format(evolution.nfe))
    # Optimal
    plt.figure("Variable space")
    plt.plot([1, 0], [0, 1])
    plt.figure("Fitness space")
    plt.plot([np.sqrt(2), 0], [0, np.sqrt(2)])
    # Evaluations
    for problem in evolution.evaluations:
        problem.logger.plot_evaluations()
    # Front
    evolution.plot_result(evolution.archive)
    plt.show()


def main2():
    """Main2"""
    n_evaluations = 1000
    problem = SalamanderEvolution(n_evaluations)
    with pla.MultiprocessingEvaluator() as evaluator:
        algorithm = pla.NSGAIII(
            problem=problem,
            evaluator=evaluator,
            divisions_outer=20
        )
        print("Running")
        algorithm.run(n_evaluations)
        print("Done")
    problem.logger.plot_non_dominated_front(algorithm.result)
    plt.show()


def profile():
    """Profile with cProfile"""
    import cProfile
    import pstats
    cProfile.run("main()", "simulation.profile")
    pstat = pstats.Stats("simulation.profile")
    pstat.sort_stats('time').print_stats(30)
    pstat.sort_stats('cumtime').print_stats(30)


if __name__ == "__main__":
    # main2()
    # main()
    profile()
