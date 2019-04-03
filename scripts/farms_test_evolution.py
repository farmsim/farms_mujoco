#!/usr/bin/env python3
"""Test salamander evolutions"""

from farms_bullet.simulation import SalamanderSimulation
from farms_bullet.simulation_options import SimulationOptions
from farms_bullet.model_options import ModelOptions
import numpy as np
import pygmo as pg


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

    def fitness(self, decision_vector):
        """Fitnesss"""
        model_options = ModelOptions(
            frequency=decision_vector[0],
            body_stand_amplitude=decision_vector[1]
        )
        sim = SalamanderSimulation(self.simulation_options, model_options)
        print("Running for parameters:\n{}".format(decision_vector))
        sim.run()
        distance_traveled = np.linalg.norm(
            sim.experiment_logger.positions.data[-1]
            - sim.experiment_logger.positions.data[0]
        )
        torques_sum = (np.abs(sim.experiment_logger.motors.joints_cmds())).sum()
        return [torques_sum/distance_traveled]

    def get_name(self):
        """Get name"""
        return self._name

    @staticmethod
    def get_bounds():
        """Get bounds"""
        return ([0, 0], [3, 0.5])


def main():
    """Main"""
    problem = SalamanderEvolution()
    algorithm = pg.algorithm(pg.sade(gen=5))
    population = pg.population(problem, size=7)
    population = algorithm.evolve(population)
    print("Champion fitness:\n{}".format(population.champion_f))
    print("Champion vector:\n{}".format(population.champion_x))


if __name__ == '__main__':
    main()
