"""Farms multi-objective optimisation for salamander experiment"""

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
        plt.figure("Variable space")
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

    def __init__(self, problem, algorithms, **kwargs):
        super(Evolution, self).__init__()
        self.problem = problem
        self.algorithms = algorithms
        self.n_population = kwargs.pop("n_population", None)
        self.n_generations = kwargs.pop("n_generations", None)
        self.n_runs = kwargs.pop("n_runs", None)
        self.options = kwargs.pop("options", {})
        self.nfe = 0
        self.archive = pla.Archive()
        self.pool = Pool()
        self.evaluations = []

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
                        self.options
                    )
                    for algorithm in self.algorithms
                ]
            )
            for algorithm, problem, _archive, nfe in results:
                self.nfe += nfe
                print("Computing nondominated front")
                # nondominated = pla.nondominated(algorithm.result)
                print("Updating archive (n={})".format(len(_archive)))
                self.update_archive(self.archive, _archive)
                # self.update_archive(self.archive, nondominated)
                print("Archive size: {}".format(len(self.archive)))
                self.evaluations.append(problem)

    @staticmethod
    def island(algorithm, problem, n_population, n_generations, archive, options):
        """Island"""
        problem = problem(n_generations*n_population)
        nfe = 0
        for _ in range(n_generations):
            options["generator"] = pla.InjectedPopulation(archive)
            _algorithm = algorithm(
                problem,
                population_size=n_population,
                # divisions_inner=0,
                # divisions_outer=10,  # n_evaluations//10,
                # epsilons=0.05,
                archive=archive,
                **options
            )
            _algorithm.run(n_population)
            Evolution.update_archive(
                archive,
                pla.nondominated(_algorithm.result)
            )
            nfe += _algorithm.nfe
            # options["generator"] = pla.InjectedPopulation(_algorithm.result)
        return _algorithm, problem, archive, nfe

    @staticmethod
    def plot_non_dominated_fronts(result):
        """Plot nondominated front"""
        nondominated_solutions = pla.nondominated(result)
        Evolution.plot_result(nondominated_solutions)
        print("Pareto front size: {}".format(len(nondominated_solutions)))

    @staticmethod
    def plot_result(result):
        """Plot result"""

        # Fitness
        plt.figure("Fitness space")
        plt.plot(
            [s.objectives[0] for s in result],
            [s.objectives[1] for s in result],
            "ro"
        )
        plt.grid(True)

        # Variables
        plt.figure("Variable space")
        plt.plot(
            [s.variables[0] for s in result],
            [s.variables[1] for s in result],
            "ro"
        )
        plt.grid(True)

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
    """ Example problem"""

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
