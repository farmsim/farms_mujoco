"""Farms multi-objective optimisation for salamander experiment"""

import time
from multiprocessing import Pool

import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt


class SphereFunction:
    """Sphere function"""

    def __init__(self, dim):
        super(SphereFunction, self).__init__()
        self.dim = dim

    def fitness(self, x):
        """Fitness"""
        print("Fitness called")
        time.sleep(0.2)
        return [sum(x*x)]

    def batch_fitness(self, x):
        """Fitness"""
        n = len(x)//self.dim
        print("Batch ({} individuals)".format(n))
        pool = Pool(4)
        batch = np.concatenate(
            pool.map(
                self.fitness,
                [
                    x[i*self.dim:(i+1)*self.dim]
                    for i in range(n)
                ]
            )
        )
        print("Batch complete ({} fitnesses)".format(len(batch)))
        return batch

    def get_bounds(self):
        """Get bounds"""
        return ([-10, -10], [10, 10])


def main():
    """Main"""
    prob = pg.problem(SphereFunction(dim=2))
    algo = pg.algorithm(pg.pso(gen=3))
    pop = pg.population(
        prob=prob,
        size=4,
        b=pg.default_bfe()
    )
    # pop = algo.evolve(pop)
    # print(pop.champion_f)
    print("Running island")
    isl = pg.island(
        algo=algo,
        pop=pop,
        # b=pg.default_bfe(),
        udi=pg.mp_island()
    )
    isl.evolve(1)
    isl.wait()
    pop = isl.get_population()
    print("Champion fitness: {}".format(pop.champion_f))


if __name__ == '__main__':
    main()
