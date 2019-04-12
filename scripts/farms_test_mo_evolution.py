#!/usr/bin/env python3
"""Test salamander multi-objective evolutions"""

import pickle
from multiprocessing import Pool
from farms_bullet.simulation import SalamanderSimulation
from farms_bullet.simulation_options import SimulationOptions
from farms_bullet.model_options import ModelOptions
import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt


class SalamanderEvolution:
    """Salamander evolution"""

    def __init__(self):
        super(SalamanderEvolution, self).__init__()
        self.simulation_options = SimulationOptions.with_clargs(
            headless=True,
            fast=True,
            duration=2
        )
        self._name = "Salamander evolution"

    # Return number of objectives
    def get_nobj(self):
        return 2

    def fitness(self, decision_vector):
        """Fitnesss"""
        model_options = ModelOptions(
            frequency=decision_vector[0],
            body_stand_amplitude=decision_vector[1]
        )
        sim = SalamanderSimulation(self.simulation_options, model_options)
        print("Running for parameters:\n{}".format(decision_vector))
        sim.run()
        sim.end()
        distance_traveled = np.linalg.norm(
            sim.experiment_logger.positions.data[-1]
            - sim.experiment_logger.positions.data[0]
        )
        torques_sum = (
            np.abs(sim.experiment_logger.motors.joints_cmds())
        ).sum()*sim.sim_options.timestep/sim.sim_options.duration
        return [-distance_traveled, torques_sum]

    def get_name(self):
        """Get name"""
        return self._name

    @staticmethod
    def get_bounds():
        """Get bounds"""
        return ([0, 0], [3, 0.5])


class DummyProblem:
    """Salamander evolution"""

    def __init__(self):
        super(DummyProblem, self).__init__()
        self._name = "Salamander evolution"

    # Return number of objectives
    def get_nobj(self):
        return 2

    def fitness(self, decision_vector):
        """Fitnesss"""
        return [0, 0]

    def get_name(self):
        """Get name"""
        return self._name

    @staticmethod
    def get_bounds():
        """Get bounds"""
        return ([0, 0], [3, 0.5])


def evolution_island(population, algorithm):
    """Evolve population"""
    return algorithm.evolve(population)


def gen_populations(n_population):
    """Generate populations"""
    return pg.population(
        SalamanderEvolution(),
        size=n_population
    )


def save_populations(populations, gen_i):
    """Save populations"""
    for pop_i, population in enumerate(populations):
        filename = "./Results/pop_{}_gen_{}.pickle".format(
            pop_i,
            gen_i
        )
        with open(filename, "wb+") as output:
            pickle.dump(population, output)


def evolution(n_gen_out=2, n_gen_in=2, n_population=8, log_data=True):
    """Evolution"""
    pool = Pool(4)
    algorithms = (
        [
            pg.algorithm(pg.moead(
                gen=n_gen_in,
                neighbours=n_population//2,
                seed=seed
            ))
            for seed in range(2)
        ] + [
            pg.algorithm(pg.nsga2(gen=n_gen_in, seed=seed))
            for seed in range(2)
        ] + [
            pg.algorithm(pg.ihs(gen=n_gen_in, seed=seed))
            for seed in range(2)
        ]
    )
    populations = pool.map(
        gen_populations,
        [n_population for _ in algorithms]
    )
    save_populations(populations, 0)
    for gen_i in range(n_gen_out):
        populations = pool.starmap(
            evolution_island,
            [
                (pop, algo)
                for (pop, algo) in zip(populations, algorithms)
            ]
        )
        # Save data
        if log_data:
            save_populations(populations, gen_i+1)
    return populations


def plot_results(populations):
    """Plot results"""
    markers = ["C{}o".format(i%10) for i, _ in enumerate(populations)]
    print(markers)
    for i, population in enumerate(populations):
        print(population)
        fits, vectors = population.get_f(), population.get_x()
        # ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(fits)
        # pg.plot_non_dominated_fronts(population.get_f())
        plt.figure("Fitness")
        plt.plot(
            fits[:, 0], fits[:, 1],
            markers[i], label="Population_{}".format(i)
        )
        plt.figure("Decisions")
        plt.plot(
            vectors[:, 0], vectors[:, 1],
            markers[i], label="Population_{}".format(i)
        )
    plt.figure("Fitness")
    plt.xlabel("Distance [m]")
    plt.ylabel("Total torque consumption [Nm/s]")
    plt.legend()
    plt.grid(True)
    plt.figure("Decisions")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Body standing wave amplitude [rad]")
    plt.legend()
    plt.grid(True)


def plot_complete(population):
    """Plot results"""
    fits, vectors = population.get_f(), population.get_x()
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(fits)
    # pg.plot_non_dominated_fronts(population.get_f())

    markers = list(reversed(["C{}o".format(i) for i, _ in enumerate(ndf)]))
    for i, _ndf in enumerate(reversed(ndf)):
        plt.figure("NDF Fitness")
        plt.plot(fits[_ndf, 0], fits[_ndf, 1], markers[i], label=i)

        plt.figure("NDF Decisions")
        plt.plot(vectors[_ndf, 0], vectors[_ndf, 1], markers[i], label=i)

    plt.figure("NDF Fitness")
    plt.xlabel("Distance [m]")
    plt.ylabel("Total torque consumption [Nm/s]")
    plt.legend()
    plt.grid(True)

    plt.figure("NDF Decisions")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Body standing wave amplitude [rad]")
    plt.legend()
    plt.grid(True)


def study_ndf(populations):
    """Study populations"""
    results = [
        [pop.get_x(), pop.get_f()]
        for pop in populations
    ]
    final_size = sum([len(result[0]) for result in results])
    final_pop = pg.population(
        prob=DummyProblem(),
        size=final_size
    )
    individual_count = 0
    for result in results:
        for individual, decision_vector in enumerate(result[0]):
            final_pop.set_xf(
                individual_count,
                decision_vector,
                np.array(result[1][individual])
            )
            individual_count += 1
    plot_complete(final_pop)
    pg.plot_non_dominated_fronts(final_pop.get_f())


def main(load_data=False):
    """Main"""
    if not load_data:
        populations = evolution(
            n_gen_out=5,
            n_gen_in=1,
            n_population=8
        )
    else:
        populations = [
            None for pop_i in range(6)
            for gen_i in range(5)
        ]
        count = 0
        for gen_i in range(5):
            for pop_i in range(6):
                filename = "./Results/pop_{}_gen_{}.pickle".format(pop_i, gen_i)
                with open(filename, "rb") as log_file:
                    populations[count] = pickle.load(log_file)
                    count += 1
    plot_results(populations)
    study_ndf(populations)
    plt.show()


if __name__ == '__main__':
    main()
