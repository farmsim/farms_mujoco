"""Archipelago evolution"""

import time

import pygmo as pg
import numpy as np


class ArchiEvolution:
    """ArchiEvolution"""

    def __init__(self, problem, algorithms, n_pop, n_gen, **kwargs):
        super(ArchiEvolution, self).__init__()
        # self.problem = problem
        self.problem = problem
        self._problem = pg.problem(problem)
        self.algorithms = [
            pg.algorithm(algorithm)
            for algorithm in algorithms
        ]
        # self.n_pop = n_pop
        # self.n_gen = n_gen
        self.n_isl = len(algorithms)
        self.pops = [
            [None for _ in range(n_gen)]
            for _ in range(self.n_isl)
        ]
        for j_isl in range(self.n_isl):
            self.pops[j_isl][0] = pg.population(self._problem, size=n_pop)
        print("Running problem: {}".format(self._problem.get_name()))
        self.islands = [
            pg.island(
                algo=algorithm,
                pop=pop[0],
                udi=pg.mp_island()
            )
            for algorithm, pop in zip(self.algorithms, self.pops)
        ]
        self._migration = kwargs.pop("migration", False)
        self.evolve()

    @property
    def n_gen(self):
        """Number of generations"""
        return len(self.pops[0])-1

    def evolve(self):
        """Evolve"""
        print("  Running evolution", end="", flush=True)
        tic = time.time()
        for gen, _ in enumerate(self.pops[0][:-1]):
            for i_isl, isl in enumerate(self.islands):
                isl.evolve()
            for i_isl, isl in enumerate(self.islands):
                isl.wait()
            # Save population
            for i_isl, isl in enumerate(self.islands):
                self.pops[i_isl][gen+1] = isl.get_population()
            # Migrate
            if self._migration:
                if not gen % 10:
                    self._migration.apply(self.pops, gen+1)
            # self.pops[gen+1] = self.algorithm.evolve(self.pops[gen])
        toc = time.time()
        print(" (time: {} [s])".format(toc-tic))
        print("  Number of evaluations: {}".format(
            self.pops[0][-1].problem.get_fevals()
        ))

    def champion(self):
        """Best solution"""
        champions = self.champions()
        return champions[np.argmin(champions[:, 1])]

    def champions(self):
        """Best solutions"""
        return np.array([
            [pop[-1].champion_x, pop[-1].champion_f]
            for pop in self.pops
        ])
