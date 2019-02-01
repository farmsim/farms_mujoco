#!/usr/bin/env python3
"""Salamander - Test Pagmo2 evolution"""

import time
from multiprocessing import Pool

import pygmo as pg
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm


class SphereFunction:
    """SphereFunction"""

    @staticmethod
    def fitness_function_fast(decision_vector):
        """Fitnesss"""
        return [np.linalg.norm(decision_vector - SphereFunction.best_known())]

    @staticmethod
    def fitness_function(decision_vector):
        """Fitnesss"""
        time.sleep(0.5)
        return SphereFunction.fitness_function_fast(decision_vector)

    def fitness(self, decision_vector):
        """Fitnesss"""
        print("Computing for {}".format(decision_vector))
        return self.fitness_function(decision_vector)

    @staticmethod
    def get_bounds():
        """Get bounds"""
        return ([-1, -1], [1, 1])

    @staticmethod
    def best_known():
        """Best known"""
        return np.array([1, 2])


def run_archipelago():
    """ Run archipelago"""
    algo = pg.algorithm(pg.cmaes(gen=3, force_bounds=True))
    prob = pg.problem(SphereFunction())
    archi = pg.archipelago(n=10, algo=algo, prob=prob, pop_size=5)
    print("\nRUNNING EVOLUTION\n")
    archi.evolve()
    while(archi.status == pg.evolve_status.busy):
        print("Status: {}".format(archi.status))
        time.sleep(0.5)
    print(archi.status)


class JonIsland:
    """Island"""

    def run_evolve(self, algo, pop):
        print("Hello")
        return algo, pop


def sort_population(pop, verbose=False):
    """Sort population"""
    _xf = np.concatenate([pop.get_x(), pop.get_f()], axis=1)
    _xf_sorted = _xf[_xf[:, -1].argsort()]
    for _i, _xf in enumerate(_xf_sorted):
        if verbose:
            print("i: {} x: {} f:{}".format(_i, _x, _f))
        _x = _xf[:-1]
        _f = [_xf[-1]]
        pop.set_xf(_i, _x, _f)
    return pop, _xf_sorted


class JonAlgorithm:
    """JonAlgorithm"""

    def __init__(self, gen, n_pool=10):
        super(JonAlgorithm, self).__init__()
        self._gen = gen
        self._n_pool = n_pool

    def evolve(self, pop):
        """Evolve"""
        pool = Pool(10)
        print(pop)
        proba = np.arange(10) + 1
        proba = np.flip(proba/np.sum(proba)).tolist()
        best_f = pop.champion_f
        for gen in range(self._gen):
            print("Generation {}".format(gen))
            # Sort population
            pop, _xf = sort_population(pop)
            # Select the best
            decisions = (
                _xf[np.random.choice(len(proba), 5, p=proba), :-1]
                + np.random.normal(0, 0.1, [5, 2])
            )
            # print("Computing fitnesses")
            fitnesses = pool.map(
                SphereFunction.fitness_function,
                decisions
            )
            for dec, fit in zip(decisions, fitnesses):
                # print("JonDecision: {} Fitness : {}".format(dec, fit))
                pop.push_back(x=dec, f=fit)
            if pop.champion_f < best_f:
                best_f = pop.champion_f
                print("NEW BEST FOUND: {}".format(best_f))
        print("Best: x={} f={}".format(pop.champion_x, pop.champion_f))
        return pop


def plot_fitness(problem, distribution, best=None, figure=None):
    """Plot fitess landscape"""
    # x_distribution, y_distribution = (
    #     [
    #         np.arange(-3.0, 3.0, delta),
    #         np.arange(-3.0, 3.0, delta)
    #     ] if distribution is None else distribution
    # )
    # _x, _y = np.meshgrid(distribution[0], distribution[1])
    # _z = np.array([problem(np.array([_i, _j])) for _i, _j in zip(_x, _y)])
    if figure is not None:
        plt.figure(figure)
    _z = np.array([
        [
            problem(np.array([_x, _y]))[0]
            for _y in distribution[1]
        ] for _x in distribution[0]
    ])
    plt.imshow(
        _z.transpose(), interpolation='bicubic',  # bicubic
        cmap=cm.RdYlGn,
        origin='lower',
        extent=[
            min(distribution[0]), max(distribution[0]),
            min(distribution[1]), max(distribution[1])
        ],
        # vmax=abs(_z).max(), vmin=-abs(_z).max()
    )
    plt.colorbar()
    if best is not None:
        plt.plot(best[0], best[1], "b*", markersize=20)

    # ln, = plt.plot([], [], 'ro', animated=True)

    # def init():
    #     ax.set_xlim(0, 2*np.pi)
    #     ax.set_ylim(-1, 1)
    #     return ln,

    # def update(frame):
    #     xdata.append(frame)
    #     ydata.append(np.sin(frame))
    #     ln.set_data(xdata, ydata)
    #     return ln,

    # return FuncAnimation(
    #     fig, update,
    #     frames=np.linspace(0, 2*np.pi, 128),
    #     init_func=init, blit=True
    # )



def main():
    """Main"""
    # Problem
    prob = pg.problem(SphereFunction())
    # Algorithm
    # algo = pg.algorithm(pg.cmaes(gen=10))
    algo = pg.algorithm(JonAlgorithm(gen=10))
    # Population
    n_pop = 10
    pop = pg.population(prob)
    decisions = np.random.ranf([n_pop, 2])
    p = Pool(10)
    fitnesses = p.map(SphereFunction.fitness_function, decisions)
    for dec, fit in zip(decisions, fitnesses):
        print("Decision: {} Fitness : {}".format(dec, fit))
        pop.push_back(x=dec, f=fit)
    # Evolution
    print("\nEVOLUTION\n")
    pop = algo.evolve(pop)
    # isl = pg.island(algo=algo, prob=prob, size=0, udi=pg.mp_island())
    # isl.set_population(pop)
    # isl.evolve()
    # while isl.status == pg.evolve_status.busy:
    #     print("isl.status: {}".format(isl.status))
    #     print("Population:\n{}".format(isl.get_population()))
    #     time.sleep(0.5)
    # print("Population:\n{}".format(isl.get_population()))


if __name__ == '__main__':
    for problem in [
            pg.ackley(dim=2),
            pg.griewank(dim=2),
            # pg.hock_schittkowsky_71(dim=2),
            # pg.inventory(dim=2),
            # pg.luksan_vlcek1(dim=2),
            pg.rastrigin(dim=2),
            # pg.minlp_rastrigin(dim=2),
            pg.rosenbrock(dim=2),
            pg.schwefel(dim=2)
    ]:
        _problem = pg.problem(problem)
        print("Running problem: {}".format(_problem.get_name()))
        bounds_min, bounds_max = _problem.get_bounds()
        algo = pg.algorithm(pg.cmaes(gen=10, force_bounds=True))
        pop = pg.population(_problem, size=100)
        pop = algo.evolve(pop)
        plot_fitness(
            _problem.fitness,
            distribution=[
                np.linspace(bounds_min[0], bounds_max[0], 300),
                np.linspace(bounds_min[1], bounds_max[1], 300)
            ],
            best=problem.best_known(),
            figure=_problem.get_name()
        )
    plt.show()
