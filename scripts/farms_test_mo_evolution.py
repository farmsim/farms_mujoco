#!/usr/bin/env python3
"""Test salamander multi-objective evolutions"""

from multiprocessing import Pool
from farms_bullet.simulation import Simulation
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
        sim = Simulation(self.simulation_options, model_options)
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


def evolution_island(population=None, seed=0):
    """Evolve population"""
    problem = SalamanderEvolution()
    algorithm = pg.algorithm(pg.moead(gen=1, neighbours=5, seed=seed))
    if not population:
        population = pg.population(problem, size=10)
    return algorithm.evolve(population)


def evolution(n_generations=2):
    """Evolution"""
    n_generations = 2
    pool = Pool(4)
    populations = [None for _ in range(4)]
    for _ in range(n_generations):
        populations = pool.starmap(
            evolution_island,
            [
                (pop, i)
                for i, pop in enumerate(populations)
            ]
        )
    return populations


def plot_results(populations):
    """Plot results"""
    markers = ["r.", "g.", "b.", "k."]
    for i, population in enumerate(populations):
        print(population)
        fits, vectors = population.get_f(), population.get_x()
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(fits)
        print(ndf)
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
    plt.show()


def main():
    """Main"""
    populations = evolution(n_generations=2)
    plot_results(populations)


if __name__ == '__main__':
    main()
