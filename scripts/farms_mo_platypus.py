"""Farms multi-objective optimisation for salamander experiment"""

import platypus as pla
import matplotlib.pyplot as plt

from farms_bullet.evolution.evolution import Evolution, ExampleProblem


def main():
    """Main"""
    # algorithms = [
    #     pla.NSGAII,
    #     (pla.NSGAIII, {"divisions_outer":12}),
    #     (pla.CMAES, {"epsilons":[0.05]}),
    #     pla.GDE3,
    #     pla.IBEA,
    #     (pla.MOEAD, {
    #         # "weight_generator": normal_boundary_weights,
    #         "divisions_outer":12
    #     }),
    #     (pla.OMOPSO, {"epsilons":[0.05]}),
    #     pla.SMPSO,
    #     pla.SPEA2,
    #     (pla.EpsMOEA, {"epsilons":[0.05]})
    # ]
    n_population = 10  # int(1e2)
    n_generations = 5  # int(1e2)
    evolution = Evolution(
        problem=ExampleProblem,
        algorithms=[
            pla.NSGAII,
            pla.CMAES,
            pla.GDE3,
            pla.SMPSO,
            # pla.IBEA,
            # pla.SPEA2
        ],
        n_population=n_population,
        n_generations=n_generations,
        n_runs=3
    )
    evolution.run_evolution()
    print("Evolution complete")
    print("Number of evaluations: {}".format(evolution.nfe))
    evolution.plot_non_dominated_fronts(evolution.archive)
    # plt.xlim([-0.5, 1.5])
    # plt.ylim([-0.5, 1.5])
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
    # main()
    profile()
