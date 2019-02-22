"""Evolution viewer"""

import time

import pygmo as pg

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.colors import LogNorm

from .archipelago import ArchiEvolution
from .benchmark_algorithms import QuadraticFunction


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

    def run_evolutions(self, migration=None):
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
                ax=self.axes[i],
                migration=migration
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


def plot_fitness(axe, problem, distribution, best=None, log=False):
    """Plot fitess landscape"""
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
