""" Writer """

import time

from salamander_generation import generate_walking
import numpy as np

from communication import Communication, MPIsettings
from compute_fitness import compute_fitness


class Individual(dict):
    """ Individual """

    STATUS = ["pending", "in_simulation", "complete"]

    def __init__(self, name):
        super(Individual, self).__init__()
        self["name"] = name
        self["parameters"] = None
        self["status"] = None
        self["fitness"] = None

    @property
    def name(self):
        """ Name """
        return self["name"]

    @name.setter
    def name(self, value):
        """ Name """
        self["name"] = value

    @property
    def parameters(self):
        """ Parameters """
        return self["parameters"]

    @parameters.setter
    def parameters(self, value):
        """ Parameters """
        self["parameters"] = value

    @property
    def status(self):
        """ Status """
        return self["status"]

    @status.setter
    def status(self, value):
        """ Status """
        if value in self.STATUS:
            self["status"] = value
        else:
            raise Exception("Status '{}' is not a valid status".format(value))

    @property
    def fitness(self):
        """ Fitness """
        return self["fitness"]

    def generate(self):
        """ Generate individual """
        generate_walking(self.name, self.parameters[0])

    def compute_fitness(self):
        """ Compute fitness of individual """
        self["fitness"] = compute_fitness(
            ".gazebo/models/{}".format(self.name),
            "link_body_0"
        )
        return self["fitness"]


class Population(list):
    """ Population to be spawned """

    def __init__(self, num=0):
        super(Population, self).__init__()
        if num:
            self.extend([
                Individual("salamander_{}".format(i))
                for i in range(num)
            ])

    def __str__(self):
        """ Individuals """
        return "{}: {}\n{}: {}\n{}: {}".format(
            "Individuals left",
            self.individuals_pending,
            "Individuals simulating",
            self.individuals_simulating,
            "Individuals simulated",
            self.individuals_simulated
        )

    @property
    def individuals_pending(self):
        """ Individuals_pending """
        return sum([
            1
            for individual in self
            if individual.status == "pending"
        ])

    @property
    def individuals_simulating(self):
        """ Individuals_simulating """
        return sum([
            1
            for individual in self
            if individual.status == "in_simulation"
        ])

    @property
    def individuals_simulated(self):
        """ Individuals_simulated """
        return sum([
            1
            for individual in self
            if individual.status == "complete"
        ])


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
        n_individuals = 5
        self.pop = Population(n_individuals)
        for i, freq in enumerate(np.linspace(0, 2, n_individuals)):
            name = "salamander_{}".format(float(freq)).replace(".", "d")
            print("Generating {} with frequency {}".format(name, freq))
            self.pop[i].name = name
            self.pop[i].parameters = [float(freq)]
            self.pop[i].status = "pending"
            self.pop[i].generate()
            # self.pop.add_individuals([name])

    def run(self):
        """ Run evolution """
        # Messaging
        self.comm.init_send_individuals(self.pop)
        print("Evolver: All data has been sent, now waiting")
        while self.pop.individuals_pending + self.pop.individuals_simulating:
            print(
                (
                    "Evolver: Individuals_pending: {}"
                    ", individuals_simulating: {}"
                ).format(
                    self.pop.individuals_pending,
                    self.pop.individuals_simulating
                )
            )
            self.comm.check_receive(self.pop)
            # COMPUTE FITNESS
            for individual in self.pop:
                if individual.status == "complete" and not individual.fitness:
                    fitness = individual.compute_fitness()
                    print("Fitness for {}: {}".format(individual, fitness))
            time.sleep(0.1)

        print("Evolver: Closing")
        for i in range(self.mpi.size-1):
            self.mpi.comm.send("close", dest=i+1, tag=0)
        print("Final scores:")
        for individual in self.pop:
            print("Individual {}: {}".format(
                individual.parameters,
                individual.fitness
            ))


if __name__ == '__main__':
    EVOLVER = Evolver()
    EVOLVER.run()
