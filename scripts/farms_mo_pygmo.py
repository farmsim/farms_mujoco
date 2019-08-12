"""Farms multi-objective optimisation for salamander experiment"""

# import time
from multiprocessing import Pool

import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt


PARALLEL = False  # False


class SphereFunction:
    """Sphere function"""

    def __init__(self, dim, n_evaluations):
        super(SphereFunction, self).__init__()
        self.dim = dim
        self.evaluations_x = np.zeros([n_evaluations, self.dim])
        self.evaluations_f = np.zeros([n_evaluations, self.get_nobj()])
        self.evaluations_x[:], self.evaluations_f[:] = np.nan, np.nan
        self.iteration = 0

    @staticmethod
    def get_nobj():
        """Get number of objectives"""
        return 2

    # def log_fitness(fitness_function):
    #     """Log fitness decorator"""
    #     def inner(self, variables):
    #         fitness = fitness_function(self, variables)
    #         print("Saving evaluation (n={})".format(len(self.evaluations)))
    #         self.evaluations.append([variables, fitness])
    #         return fitness
    #     return inner

    # @log_fitness
    def fitness(self, variables):
        """Fitness"""
        # time.sleep(1e-2)
        fitness = self.compute_fitness(variables)
        print("Saving evaluation (n={})".format(len(self.evaluations_x)))
        self.evaluations_x[self.iteration] = variables
        self.evaluations_f[self.iteration] = fitness
        self.iteration += 1
        return fitness

    @staticmethod
    def compute_fitness(variables):
        """Fitness"""
        return (
            np.sqrt(
                + (variables[0]-0)**2
                + (variables[1]-1)**2
            ),
            np.sqrt(
                + (variables[0]-1)**2
                + (variables[1]-0)**2
            )
        )

    def batch_fitness(self, variables):
        """Fitness"""
        n_objs = len(variables)//self.dim
        print("Batch ({} individuals)".format(n_objs))
        pool = Pool()
        batch = np.concatenate(
            pool.map(
                self.fitness,
                [
                    variables[i*self.dim:(i+1)*self.dim]
                    for i in range(n_objs)
                ]
            )
        )
        print("Batch complete ({} fitnesses)".format(np.shape(batch)))
        pool.close()
        return batch

    def get_bounds(self):
        """Get bounds"""
        return (
            [-10 for _ in range(self.dim)],
            [10 for _ in range(self.dim)]
        )


def plot_non_dominated_fronts(points, marker='o', comp=[0, 1], **kwargs):
    """Based on Pygmo

    Plots the nondominated fronts of a set of points.  Makes use of
    :class:`~pygmo.fast_non_dominated_sorting` to compute the non dominated
    fronts.

    Args: points (2d array-like): points to plot marker (``str``): matplotlib
    marker used to plot the *points* comp (``list``): Components to be
    considered in the two dimensional plot (useful in many-objectives cases)

    Returns: ``matplotlib.axes.Axes``: the current ``matplotlib.axes.Axes``
    instance on the current figure

    Examples: >>> from pygmo import * >>> prob = problem(zdt()) >>> pop =
    population(prob, 40) >>> ax = plot_non_dominated_fronts(pop.get_f()) #
    doctest: +SKIP
    """
    fronts, _, _, _ = pg.fast_non_dominated_sorting(points)

    # We define the colors of the fronts (grayscale from black to white)
    cl = list(zip(np.linspace(0.1, 0.9, len(fronts)),
                  np.linspace(0.1, 0.9, len(fronts)),
                  np.linspace(0.1, 0.9, len(fronts))))

    # fig, ax = plt.subplots(**kwargs)
    ax = plt.gca()

    for ndr, front in enumerate(fronts):
        # We plot the points
        for idx in front:
            ax.plot(
                points[idx][comp[0]],
                points[idx][comp[1]],
                marker=marker,
                color=cl[ndr]
            )
        # We plot the fronts
        # Frist compute the points coordinates
        x = [points[idx][0] for idx in front]
        y = [points[idx][1] for idx in front]
        # Then sort them by the first objective
        tmp = [(a, b) for a, b in zip(x, y)]
        tmp = sorted(tmp, key=lambda k: k[0])
        # Now plot using step
        ax.step(
            [c[0] for c in tmp],
            [c[1] for c in tmp],
            color=cl[ndr],
            where='post'
        )

    return ax


def main():
    """Main"""
    dim = 2
    n_pop = 20
    n_gen = 100
    n_inter = 1
    prob = SphereFunction(dim=dim, n_evaluations=n_pop*n_inter*(n_gen+1))
    # prob = pg.problem(_prob)
    print("Setting population")
    if PARALLEL:
        bfe = pg.bfe()
    pop = pg.population(
        prob=prob,
        size=n_pop,
        b=bfe if PARALLEL else None
    )
    print("Setting algorithm")
    _algo = pg.nsga2(  # pg.moead(  # pg.nsga2(
        gen=n_inter,
        # neighbours=dim  # len(pop)-1
    )
    if PARALLEL:
        _algo.set_bfe(bfe)
    algo = pg.algorithm(_algo)
    hypervolume = np.zeros(n_gen)
    refpoint = [1e3, 1e3]
    print("Running optimisation")
    for generation in range(n_gen):
        print("Generation: {}/{}".format(generation+1, n_gen))
        pop = algo.evolve(pop)
        hypervolume[generation] = pg.hypervolume(pop).compute(refpoint)
    print("Population:\n{}".format(pop))

    # from IPython import embed; embed()
    prob = pop.problem.extract(SphereFunction)
    evaluations_d = np.array([
        evaluation
        for evaluation
        in prob.evaluations_x
    ])
    evaluations_f = np.array([
        evaluation
        for evaluation
        in prob.evaluations_f
    ])

    print("Evaluations: {}".format(prob.iteration))

    # Optimal
    plt.figure("Decisions")
    plt.plot([1, 0], [0, 1])
    plt.figure("Fitnesses")
    plt.plot([np.sqrt(2), 0], [0, np.sqrt(2)])
    # Plot decisions
    decisions = pop.get_x()
    plt.figure("Decisions")
    if not PARALLEL:
        plt.plot(evaluations_d[:, 0], evaluations_d[:, 1], "bo")
    plt.plot(decisions[:, 0], decisions[:, 1], "ro")
    plt.grid(True)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    # Plot fitnesses
    # fitnesses = pop.get_f()
    plt.figure("Fitnesses")
    if not PARALLEL:
        plt.plot(evaluations_f[:, 0], evaluations_f[:, 1], "bo")
    plot_non_dominated_fronts(pop.get_f(), comp=[0, 1])
    plt.grid(True)
    # Hypervolume
    plt.figure("Hypervolume")
    plt.plot(np.prod(refpoint) - hypervolume)
    plt.xlabel("Generations")
    plt.ylabel("Hypervolume")
    plt.grid(True)
    # Show
    plt.show()


if __name__ == '__main__':
    main()
