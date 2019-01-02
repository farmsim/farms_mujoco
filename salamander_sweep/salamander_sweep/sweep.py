""" Salamander sweep """

import time

from .communication import Communication, MPIsettings
from .compute_fitness import compute_fitness


class Individual(dict):
    """ Individual """

    STATUS = ["pending", "in_simulation", "complete"]

    def __init__(self, name):
        super(Individual, self).__init__()
        self["name"] = name
        self["status"] = "pending"
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

    def compute_fitness(self):
        """ Compute fitness of individual """
        self["fitness"] = compute_fitness(
            ".gazebo/models/{}".format(self.name),
            "link_body_0"
        )
        return self["fitness"]


class Population(list):
    """ Population to be spawned """

    def __init__(self, names):
        super(Population, self).__init__()
        self.extend([Individual(name) for name in names])

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


class Sweep:
    """ Sweep """

    def __init__(self, models):
        super(Sweep, self).__init__()
        self.mpi = MPIsettings()
        print("Sweep is running (rank={}, size={})".format(
            self.mpi.rank,
            self.mpi.size
        ))
        assert self.mpi.rank == 0, (
            "Rank of sweep must be 0, but is {}".format(self.mpi.rank)
        )
        self.comm = Communication(self.mpi)
        self.pop = Population(models)

    def run(self):
        """ Run evolution """
        # Messaging
        self.comm.init_send_individuals(self.pop)
        print("Sweep: All data has been sent, now waiting")
        while self.pop.individuals_pending + self.pop.individuals_simulating:
            self.loop()
        print((
            "Sweep complete:"
            "\n    Pending:    {}"
            "\n    Simulating: {}"
            "\n    Simulated:  {}"
        ).format(
            self.pop.individuals_pending,
            self.pop.individuals_simulating,
            self.pop.individuals_simulated
        ))
        self.close()

    def loop(self):
        """ Loop """
        print(
            (
                "Sweep: Individuals_pending: {}"
                ", individuals_simulating: {}"
            ).format(
                self.pop.individuals_pending,
                self.pop.individuals_simulating
            )
        )
        self.comm.check_receive(self.pop)
        completion = False
        for individual in self.pop:
            if individual.status == "complete" and not individual.fitness:
                fitness = individual.compute_fitness()
                print("Fitness for {}: {}".format(individual.name, fitness))
                completion = True
        if not completion:
            time.sleep(0.1)

    def close(self):
        """ Close all """
        print("Sweep: Closing")
        for i in range(self.mpi.size-1):
            self.mpi.comm.send("close", dest=i+1, tag=0)
        print("Final scores:")
        for individual in self.pop:
            print("{}: {}".format(
                individual.name,
                individual.fitness
            ))
