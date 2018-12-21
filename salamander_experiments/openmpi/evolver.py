""" Writer """

import time
from mpi4py import MPI

from communication import Communication, MPIsettings
from compute_fitness import compute_fitness
from salamander_generation import generate_walking
import numpy as np


class Individual:
    """ Individual """

    def __init__(self, name):
        super(Individual, self).__init__()
        self._status = None
        self._name = name

    @property
    def name(self):
        """ Name """
        return self._name

    @property
    def status(self):
        """ Status """
        return self._status


class Population(list):
    """ Population to be spawned """

    def __init__(self):
        super(Population, self).__init__()
        self._individuals_left = 0
        self._individuals_simulating = 0
        self._individuals_simulated = 0

    def __str__(self):
        """ Individuals """
        return "{}: {}\n{}: {}\n{}: {}".format(
            "Individuals left",
            self._individuals_left,
            "Individuals simulating",
            self._individuals_simulating,
            "Individuals simulated",
            self._individuals_simulated
        )

    @property
    def individuals_left(self):
        """ Individuals_left """
        return self._individuals_left

    @property
    def individuals_simulating(self):
        """ Individuals_simulating """
        return self._individuals_simulating

    @property
    def individuals_simulated(self):
        """ Individuals_simulated """
        return self._individuals_simulated

    def add_individuals(self, individuals):
        """ Add individuals """
        self.extend(individuals)
        self._individuals_left = len(self)

    def consume(self):
        """ Consume individual """
        self._individuals_left -= 1
        self._individuals_simulating += 1
        self._individuals_simulated += 1

    def simulation_complete(self):
        """ Simulation complete for individual """
        self._individuals_simulating -= 1


class Evolver:
    """ Evolver """

    def __init__(self):
        super(Evolver, self).__init__()
        self.mpi = MPIsettings()
        print("Evolver is running (rank={}, size={})".format(
            self.mpi.rank,
            self.mpi.size
        ))
        assert self.mpi.rank == 0, (
            "Rank of evolver must be 0, but is {}".format(self.mpi.rank)
        )
        self.comm = Communication(self.mpi)
        self.pop = Population()
        for freq in np.linspace(0, 2, 5):
            name = "salamander_{}".format(float(freq)).replace(".", "d")
            print("Generating {} with frequency {}".format(name, freq))
            generate_walking(name, freq)
            self.pop.add_individuals([name])

    def run(self):
        """ Run evolution """
        # Messaging
        self.comm.init_send_individuals(self.pop)
        print("Evolver: All data has been sent, now waiting")
        while self.pop.individuals_left + self.pop.individuals_simulating:
            print(
                (
                    "Evolver: Individuals_left: {}, individuals_simulating: {}"
                ).format(
                    self.pop.individuals_left,
                    self.pop.individuals_simulating
                )
            )
            self.comm.check_receive(self.pop)
            # COMPUTE FITNESS
            # individual = pop.individuals[pop.individuals_simulated-1]
            # fitness = compute_fitness(
            #     ".gazebo/models/"+individual,
            #     "link_body_0"
            # )
            # print("Fitness for {}: {}".format(individual, fitness))
            time.sleep(0.1)

        print("Evolver: Closing")
        for i in range(self.mpi.size-1):
            self.mpi.comm.send("close", dest=i+1, tag=0)


if __name__ == '__main__':
    EVOLVER = Evolver()
    EVOLVER.run()
