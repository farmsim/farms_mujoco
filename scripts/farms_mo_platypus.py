"""Farms multi-objective optimisation for salamander experiment"""

# import time
from multiprocessing import Pool

import numpy as np
import platypus as pla
import matplotlib.pyplot as plt


class ProblemLogger:
    """Problem population logger"""

    def __init__(self, n_evaluations, n_vars, n_objs):
        super(ProblemLogger, self).__init__()
        self.n_evaluations = n_evaluations
        self.variables = np.zeros([n_evaluations, n_vars])
        self.objectives = np.zeros([n_evaluations, n_objs])
        self.iteration = 0

    def log(self, variables, objectives):
        """Log individual"""
        if self.iteration < self.n_evaluations:
            self.variables[self.iteration] = variables
            self.objectives[self.iteration] = objectives
        self.iteration += 1

    def plot_non_dominated_front(self, result):
        """Plot variables"""
        nondominated = pla.nondominated(result)
        plot_non_dominated_fronts(nondominated)
        print("Pareto front size: {}/{}".format(
            len(nondominated),
            self.iteration
        ))

    def plot_evaluations(self):
        """Plot variables"""

        # for solution in nondominated_solutions:
        #     print("Decision vector: {} Fitness: {}".format(
        #         solution.variables,
        #         list(solution.objectives)
        #     ))

        # Variables
        plt.figure("Varable space")
        plt.plot(
            self.variables[:, 0],
            self.variables[:, 1],
            "bo"
        )

        # Fitness
        plt.figure("Fitness space")
        plt.plot(
            self.objectives[:, 0],
            self.objectives[:, 1],
            "bo"
        )


def plot_non_dominated_fronts(result):
    """Plot nondominated front"""
    nondominated_solutions = pla.nondominated(result)

    # Fitness
    plt.figure("Fitness space")
    plt.plot(
        [s.objectives[0] for s in nondominated_solutions],
        [s.objectives[1] for s in nondominated_solutions],
        "ro"
    )
    plt.grid(True)

    # Variables
    plt.figure("Varable space")
    plt.plot(
        [s.variables[0] for s in nondominated_solutions],
        [s.variables[1] for s in nondominated_solutions],
        "ro"
    )
    plt.grid(True)

    print("Pareto front size: {}".format(
        len(nondominated_solutions)
    ))


class Schaffer(pla.Problem):

    def __init__(self, n_evaluations):
        n_vars, n_objs = 2, 2
        super(Schaffer, self).__init__(nvars=n_vars, nobjs=n_objs)
        self.types[0] = pla.Real(-10, 10)
        self.types[1] = pla.Real(-10, 10)
        self.logger = ProblemLogger(n_evaluations, n_vars, n_objs)

    def evaluate(self, solution):
        # print("Evaluating {}".format(self.logger.iteration))
        # solution.objectives[0] = (
        #     + (solution.variables[0]-4)**2
        #     + (solution.variables[0]-2)**3
        #     + (solution.variables[1]-7)**4
        # )
        # solution.objectives[1] = (
        #     + (solution.variables[0]-7)**2
        #     + (solution.variables[1]-0)**2
        # )

        solution.objectives[0] = (
            + (solution.variables[0]-0)**2
            + (solution.variables[1]-1)**2
        )
        solution.objectives[1] = (
            + (solution.variables[0]-1)**2
            + (solution.variables[1]-0)**2
        )

        # solution.objectives[0] = (
        #     (solution.variables[0] - solution.variables[1])**2
        # )
        # solution.objectives[1] = (
        #     (solution.variables[0])**2
        # )
        # solution.objectives[2] = (
        #     (solution.variables[1])**2
        # )

        self.logger.log(solution.variables, solution.objectives)


def island(algorithm, problem, n_population, n_generations, archive, options):
    """Island"""
    algorithm = algorithm(
        problem,
        population_size=n_population,
        # divisions_inner=0,
        # divisions_outer=10,  # n_evaluations//10,
        # epsilons=0.05,
        archive=archive,
        **options
    )
    for _ in range(n_generations):
        algorithm.run(n_population)
    return algorithm, problem


def update_archive(archive, solutions, rtol=1e-8, atol=1e-8):
    """Update archive"""
    for solution in solutions:
        in_archive = False
        for _archive in archive:
            if np.allclose(
                    solution.variables,
                    _archive.variables,
                    rtol=rtol,
                    atol=atol
            ):
                in_archive = True
                break
        if not in_archive:
            archive += solution


def main():
    """Main"""

    algorithms = [
        pla.NSGAII,
        (pla.NSGAIII, {"divisions_outer":12}),
        (pla.CMAES, {"epsilons":[0.05]}),
        pla.GDE3,
        pla.IBEA,
        (pla.MOEAD, {
            # "weight_generator": normal_boundary_weights,
            "divisions_outer":12
        }),
        (pla.OMOPSO, {"epsilons":[0.05]}),
        pla.SMPSO,
        pla.SPEA2,
        (pla.EpsMOEA, {"epsilons":[0.05]})
    ]

    pool = Pool()
    archive = pla.Archive()
    n_population = 10  # int(1e2)
    n_generations = 10  # int(1e2)
    n_runs = 10  # int(1e2)
    problem = Schaffer(n_generations*n_population)
    nfe = 0
    archive = pla.Archive()
    for generation in range(n_runs):
        print("Generation {}".format(generation))
        results = pool.starmap(
            island,
            [
                (
                    pla.GDE3,
                    problem,
                    n_population,
                    n_generations,
                    archive,
                    {
                        # "divisions_outer": 10
                    }
                )
                for _ in range(4)
            ]
        )
        print("Updating archive")
        for algorithm, _problem in results:
            nfe += algorithm.nfe
            print("Computing nondominated front")
            nondominated = pla.nondominated(algorithm.result)
            print("Updating archive (n={})".format(len(nondominated)))
            update_archive(archive, nondominated)
            print("Archive size: {}".format(len(archive)))
    print("Evolution complete")
    print("Number of evaluations: {}".format(nfe))
    plot_non_dominated_fronts(archive)
    plt.xlim([-0.5, 1.5])
    plt.ylim([-0.5, 1.5])
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
