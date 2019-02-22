"""Archipelago evolution"""

import time
import pygmo as pg


class ArchiEvolution:
    """ArchiEvolution"""

    def __init__(self, problem, algorithms, n_pop, n_gen, **kwargs):
        super(ArchiEvolution, self).__init__()
        self.problem = problem
        self._problem = pg.problem(self.problem)
        self.algorithms = [
            pg.algorithm(algorithm)
            for algorithm in algorithms
        ]
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.n_isl = len(algorithms)
        self.pops = [
            [None for _ in range(self.n_gen)]
            for _ in range(self.n_isl)
        ]
        for j_isl in range(self.n_isl):
            self.pops[j_isl][0] = pg.population(self._problem, size=n_pop)
        print("Running problem: {}".format(self._problem.get_name()))
        self._migrate = kwargs.pop("migrate", False)
        self.evolve()

    def evolve(self):
        """Evolve"""
        print("  Running evolution", end="", flush=True)
        tic = time.time()
        self.islands = [
            pg.island(
                algo=algorithm,
                pop=pop[0],
                udi=pg.mp_island()
            )
            for algorithm, pop in zip(self.algorithms, self.pops)
        ]
        for gen in range(self.n_gen-1):
            for i_isl, isl in enumerate(self.islands):
                isl.evolve()
            for i_isl, isl in enumerate(self.islands):
                isl.wait()
            # Save population
            for i_isl, isl in enumerate(self.islands):
                self.pops[i_isl][gen+1] = isl.get_population()
            # Migrate
            if self._migrate:
                if not gen % 10:
                    self.migrate(gen)
            # self.pops[gen+1] = self.algorithm.evolve(self.pops[gen])
        toc = time.time()
        print(" (time: {} [s])".format(toc-tic))
        print("  Number of evaluations: {}".format([
            pop.problem.get_fevals()
            for pop in self.pops[0]
        ][-1]))

    def migrate(self, gen):
        """Migrate"""
        for i_isl, _ in enumerate(self.islands[:-1]):
            worst = self.pops[i_isl+1][gen+1].worst_idx()
            self.pops[i_isl+1][gen+1].set_xf(
                worst,
                self.pops[i_isl][gen+1].champion_x,
                self.pops[i_isl][gen+1].champion_f
            )
        worst = self.pops[0][gen+1].worst_idx()
        self.pops[0][gen+1].set_xf(
            worst,
            self.pops[-1][gen+1].champion_x,
            self.pops[-1][gen+1].champion_f
        )
