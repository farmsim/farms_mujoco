"""Farms multi-objective optimisation"""

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
        Evolution.plot_non_dominated_fronts(nondominated)
        print("Pareto front size: {}/{}".format(
            len(nondominated),
            self.iteration
        ))

    def plot_evaluations(self):
        """Plot variables"""

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


class Evolution:
    """Evolution"""

    def __init__(self, problem, algorithms, n_population, n_generations, n_runs):
        super(Evolution, self).__init__()
        self.problem = problem
        self.algorithms = algorithms
        self.n_population = n_population
        self.n_generations = n_generations
        self.n_runs = n_runs
        self.nfe = 0
        self.archive = pla.Archive()
        self.pool = Pool()

    def run_evolution(self):
        """Run evolution"""
        for generation in range(self.n_runs):
            print("Generation {}".format(generation))
            results = self.pool.starmap(
                self.island,
                [
                    (
                        algorithm,
                        self.problem,
                        self.n_population,
                        self.n_generations,
                        self.archive,
                        {
                            # "divisions_outer": 10
                        }
                    )
                    for algorithm in self.algorithms
                ]
            )
            print("Updating archive")
            for algorithm, problem in results:
                self.nfe += algorithm.nfe
                print("Computing nondominated front")
                nondominated = pla.nondominated(algorithm.result)
                print("Updating archive (n={})".format(len(nondominated)))
                self.update_archive(self.archive, nondominated)
                print("Archive size: {}".format(len(self.archive)))
                problem.logger.plot_evaluations()

    @staticmethod
    def island(algorithm, problem, n_population, n_generations, archive, options):
        """Island"""
        problem = problem(n_generations*n_population)
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

    @staticmethod
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

    @staticmethod
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


class ExampleProblem(pla.Problem):

    def __init__(self, n_evaluations):
        n_vars, n_objs = 2, 2
        super(ExampleProblem, self).__init__(nvars=n_vars, nobjs=n_objs)
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


