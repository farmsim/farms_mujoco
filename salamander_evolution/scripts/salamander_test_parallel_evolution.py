#!/usr/bin/env python3
"""Salamander - Test Pagmo2 evolution"""

import time
from multiprocessing import Pool
import argparse

import pygmo as pg
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.colors import LogNorm


class QuadraticFunction:
    """QuadraticFunction"""

    def __init__(self, dim, slow=False):
        super(QuadraticFunction, self).__init__()
        self._dim = dim
        self._slow = slow
        self._name = "Quadratic Function"

    @staticmethod
    def fitness_function(decision_vector):
        """Fitnesss"""
        return [np.linalg.norm(decision_vector - QuadraticFunction.best_known())]

    def fitness(self, decision_vector):
        """Fitnesss"""
        if self._slow:
            time.sleep(0.5)
        return self.fitness_function(decision_vector)

    def get_name(self):
        """Get name"""
        return self._name

    @staticmethod
    def get_bounds():
        """Get bounds"""
        return ([-1, -1], [1, 1])

    @staticmethod
    def best_known():
        """Best known"""
        return np.array([0.5, 0.5])


def run_archipelago():
    """ Run archipelago"""
    algo = pg.algorithm(pg.cmaes(gen=3, force_bounds=True))
    prob = pg.problem(QuadraticFunction())
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
                QuadraticFunction.fitness_function,
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


def plot_fitness(axe, problem, distribution, best=None, log=False):
    """Plot fitess landscape"""
    # x_distribution, y_distribution = (
    #     [
    #         np.arange(-3.0, 3.0, delta),
    #         np.arange(-3.0, 3.0, delta)
    #     ] if distribution is None else distribution
    # )
    # _x, _y = np.meshgrid(distribution[0], distribution[1])
    # _z = np.array([problem(np.array([_i, _j])) for _i, _j in zip(_x, _y)])
    # if figure is not None:
    #     plt.figure(figure)
    _z = np.array([
        [
            problem(np.array([_x, _y]))[0]
            for _y in distribution[1]
        ] for _x in distribution[0]
    ])
    image = axe.imshow(
        _z.transpose(), interpolation='bicubic',  # bicubic
        cmap=cm.RdYlGn,
        origin='lower',
        extent=[
            min(distribution[0]), max(distribution[0]),
            min(distribution[1]), max(distribution[1])
        ],
        norm=(LogNorm() if log else None)
        # vmax=abs(_z).max(), vmin=-abs(_z).max()
    )
    plt.colorbar(image, ax=axe)
    if best is not None:
        axe.plot(best[0], best[1], "w*", markersize=20)

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


class AlgorithmViewer2D:
    """AlgorithmViewer2D"""

    def __init__(self, algorithms, n_pop, n_gen):
        super(AlgorithmViewer2D, self).__init__()

        self.algorithms = algorithms
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.problems = [
            QuadraticFunction(dim=2),
            pg.ackley(dim=2),
            pg.griewank(dim=2),
            # pg.hock_schittkowsky_71(dim=2),
            # pg.inventory(dim=2),
            # pg.luksan_vlcek1(dim=2),
            pg.rastrigin(dim=2),
            # pg.minlp_rastrigin(dim=2),
            pg.rosenbrock(dim=2),
            pg.schwefel(dim=2)
        ]
        self.viewers = [None for _, _ in enumerate(self.problems)]
        self.fig, self.axes, self.ani = None, None, None
        self.name = pg.algorithm(self.algorithms[0]).get_name()

    def run_evolutions(self):
        """Run evolutions"""
        self.fig, self.axes = plt.subplots(
            nrows=2,
            ncols=3,
            figsize=(15, 8),
            num=self.name
        )
        self.axes = np.reshape(self.axes, 6)
        for i, problem in enumerate(self.problems):
            self.viewers[i] = EvolutionViewer2D(
                problem=problem,
                algorithms=self.algorithms,
                n_pop=self.n_pop,
                n_gen=self.n_gen,
                plot_log=isinstance(problem, pg.rosenbrock),
                ax=self.axes[i]
            )

    def update_plot(self, frame):
        """Update plot"""
        return [
            item
            for viewer in self.viewers
            for item in viewer.update_plot(frame)
        ]

    def animate(self, write):
        """Animate"""
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot,
            frames=np.arange(self.n_gen-1),
            # init_func=self.init_plot,
            blit=True,
            interval=100,
            repeat=True
        )
        if write:
            writer = animation.writers['ffmpeg'](
                fps=10,
                # metadata=dict(artist=''),
                bitrate=1800
            )
            name = self.name.replace(" ", "_")
            name = name.replace(":", "_")
            filename = "{}.mp4".format(name)
            print("Saving to {}".format(filename))
            self.ani.save(filename, writer=writer)


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


class EvolutionViewer2D(ArchiEvolution):
    """EvolutionViewer2D"""

    def __init__(self, problem, algorithms, n_pop, n_gen, plot_log, **kwargs):
        super(EvolutionViewer2D, self).__init__(problem, algorithms, n_pop, n_gen, **kwargs)
        self.plot_log = plot_log
        self.pop_plots = None
        self.axe = kwargs.pop("ax", None)
        if self.axe is None:
            _ , self.axe = plt.subplots(1, 1)
        self.plot_fitness()
        self.plot_evolution()
        self.ani = None
        self.fig = plt.gcf()
        self.axe.set_title(self._problem.get_name())
        # self.animate()

    def plot_fitness(self):
        """Plot fitness"""
        print("  Plotting fitness landscape", end="", flush=True)
        tic = time.time()
        bounds_min, bounds_max = self._problem.get_bounds()
        plot_fitness(
            self.axe,
            self._problem.fitness,
            distribution=[
                np.linspace(bounds_min[0], bounds_max[0], 300),
                np.linspace(bounds_min[1], bounds_max[1], 300)
            ],
            best=(
                self.problem.best_known()
                if "best_known" in dir(self.problem)
                else None
            ),
            # figure=self._problem.get_name(),
            log=self.plot_log
        )
        toc = time.time()
        print(" (time: {} [s])".format(toc-tic))

    def plot_evolution(self):
        """Evolve"""
        print("  Plotting evolution", end="", flush=True)
        tic = time.time()
        # for gen in range(self.n_gen-1):
        #     self.plot_generation(gen)
        self.plot_generation(0)
        toc = time.time()
        print(" (time: {} [s])".format(toc-tic))

    def plot_generation(self, gen):
        """Plot population for a specific generation"""
        pops_size = len(self.pops)
        self.pop_plots = [
            self.axe.plot(
                pop[gen+1].get_x()[:, 0],
                pop[gen+1].get_x()[:, 1],
                "C{}o".format(pops_size-i)
            )[0]
            for i, pop in enumerate(self.pops)
        ]

    def update_plot(self, frame):
        """Update population plot"""
        for i, pop_plot in enumerate(self.pop_plots):
            decision_vectors = self.pops[i][frame].get_x()
            pop_plot.set_data(decision_vectors[:, 0], decision_vectors[:, 1])
        return self.pop_plots

    def animate(self, write=False):
        """Animate evolution"""
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot,
            frames=np.arange(self.n_gen-1),
            # init_func=self.init_plot,
            blit=True,
            interval=100,
            repeat=True
        )
        if write:
            writer = animation.writers['ffmpeg'](
                fps=10,
                # metadata=dict(artist=''),
                bitrate=1800
            )
            self.ani.save('{}.mp4'.format("test"), writer=writer)


def main():
    """Main"""
    # Problem
    prob = pg.problem(QuadraticFunction())
    # Algorithm
    # algo = pg.algorithm(pg.cmaes(gen=10))
    algo = pg.algorithm(JonAlgorithm(gen=10))
    # Population
    n_pop = 10
    pop = pg.population(prob)
    decisions = np.random.ranf([n_pop, 2])
    p = Pool(10)
    fitnesses = p.map(QuadraticFunction.fitness_function, decisions)
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


def parse_args():
    """ Parse arguments """
    parser = argparse.ArgumentParser(description='Test evolution')
    # parser.add_argument(
    #     'model_names',
    #     type=str,
    #     nargs='+',
    #     help='Name of Gazebo model to simulate'
    # )
    parser.add_argument(
        "-s", '--save',
        action='store_true',
        dest='save',
        help='Save results'
    )
    args = parser.parse_args()
    return args


def run_evolutions(viewer):
    """Run evolution"""
    viewer.run_evolutions()


def main2():
    """Main 2"""

    args = parse_args()
    algorithms = []
    n_threads = 8

    # Population without memory
    kwargs = {"seed": 0}
    # algorithm = pg.de(gen=1, **kwargs)
    # algorithm = pg.de(gen=1, variant=1, **kwargs)
    # algorithm = pg.sea(gen=1, **kwargs)
    # algorithm = pg.sga(gen=1, **kwargs)
    # algorithm = pg.bee_colony(gen=1, **kwargs)
    # algorithm = pg.simulated_annealing()
    # algorithm = pg.ihs(gen=1, bw_min=1e-2, **kwargs)

    algorithms.append([pg.de(gen=1, **kwargs) for _ in range(n_threads)])
    algorithms.append([pg.sea(gen=1, **kwargs) for _ in range(n_threads)])
    algorithms.append([pg.sga(gen=1, **kwargs) for _ in range(n_threads)])
    algorithms.append([pg.bee_colony(gen=1, **kwargs) for _ in range(n_threads)])

    # Population with memory
    kwargs = {"memory": True, "seed": 0}
    # algorithm = pg.cmaes(gen=1, force_bounds=True, **kwargs)
    # algorithm = pg.xnes(gen=1, force_bounds=True, **kwargs)
    # algorithm = pg.pso(gen=1, **kwargs)
    # algorithm = pg.pso_gen(gen=1, **kwargs)
    # algorithm = pg.sade(gen=1, **kwargs)
    # algorithm = pg.sade(gen=1, variant=13, **kwargs)
    # algorithm = pg.sade(gen=1, xtol=1e0, ftol=1e0, **kwargs)
    # algorithm = pg.sade(gen=1, variant=11, variant_adptv=1, **kwargs)
    # algorithm = pg.sade(gen=1, variant=2, variant_adptv=2, **kwargs)
    # algorithm = pg.de1220(gen=1, **kwargs)

    # algorithms.append([pg.pso(gen=1, **kwargs) for _ in range(n_threads)])
    # algorithms.append([pg.sade(gen=1, **kwargs) for _ in range(n_threads)])
    # algorithms.append([pg.de1220(gen=1, **kwargs) for _ in range(n_threads)])

    # kwargs = {"memory": True, "seed": 0, "force_bounds": True}
    # algorithms.append([pg.cmaes(gen=1, **kwargs) for _ in range(n_threads)])
    # algorithms.append([pg.xnes(gen=1, **kwargs) for _ in range(n_threads)])

    # Multiobjective
    # algorithm = pg.nsga2(gen=1, **kwargs)
    # algorithm = pg.moead(gen=1, **kwargs)

    # Local
    # algorithm = pg.compass_search(max_fevals=100)
    # algorithm = pg.nlopt(solver="cobyla")
    # algorithm = pg.nlopt(solver="bobyqa")
    # algorithm = pg.nlopt(solver="neldermead")

    # algorithms = []
    # kwargs = {"memory": True, "seed": 0}
    # algorithms.append([
    #     pg.sade(gen=1, variant=variant//2+1, **kwargs)
    #     for variant in range(8)
    # ])

    # Instantiate viewers
    viewers = [
        AlgorithmViewer2D(_algorithms, n_pop=10, n_gen=100)
        for _algorithms in algorithms
    ]
    print("Running islands")
    # Run evolutions
    for viewer in viewers:
        viewer.run_evolutions()
    # from multiprocessing import Pool
    # p = Pool(4)
    # print(p.map(run_evolutions, viewers))
    # Animate
    for viewer in viewers:
        viewer.animate(write=args.save)
    # Save
    if not args.save:
        plt.show()


if __name__ == '__main__':
    main2()
