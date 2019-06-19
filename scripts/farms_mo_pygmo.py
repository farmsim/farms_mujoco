"""Farms multi-objective optimisation for salamander experiment"""

# import time
from multiprocessing import Pool

import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt


class SphereFunction:
    """Sphere function"""

    def __init__(self, dim):
        super(SphereFunction, self).__init__()
        self.dim = dim

    @staticmethod
    def get_nobj():
        """Get number of objectives"""
        return 2

    @staticmethod
    def fitness(variables):
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
        return batch

    def get_bounds(self):
        """Get bounds"""
        return (
            [-10 for _ in range(self.dim)],
            [10 for _ in range(self.dim)]
        )


def main():
    """Main"""
    prob = pg.problem(SphereFunction(dim=2))
    pop = pg.population(
        prob=prob,
        size=10,
        b=pg.default_bfe()
    )
    algo = pg.algorithm(
        pg.moead(
            gen=1,
            neighbours=len(pop)//5,
        )
    )
    for generation in range(100):
        pop = algo.evolve(pop)
    print("Population:\n{}".format(pop))

    # Plot decisions
    decisions = pop.get_x()
    plt.figure("Decisions")
    plt.plot(decisions[:, 0], decisions[:, 1], "ro")
    plt.grid(True)
    plt.show()
    # Plot fitnesses
    fitnesses = pop.get_f()
    plt.figure("Fitnesses")
    plt.plot(fitnesses[:, 0], fitnesses[:, 1], "ro")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
