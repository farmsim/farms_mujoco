""" Writer """

import time
from mpi4py import MPI

from communication import MPIsettings
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


class Population:
    """ Population to be spawned """

    def __init__(self):
        super(Population, self).__init__()
        self._individuals = []
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
    def individuals(self):
        """ Individuals """
        return self._individuals

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
        self._individuals.extend(individuals)
        self._individuals_left = len(self._individuals)

    def consume(self):
        """ Consume individual """
        self._individuals_left -= 1
        self._individuals_simulating += 1
        self._individuals_simulated += 1

    def simulation_complete(self):
        """ Simulation complete for individual """
        self._individuals_simulating -= 1


class CommunicationCPP:
    """ Communication """

    def __init__(self, mpi):
        super(Communication, self).__init__()
        self.mpi = mpi
        self.status = MPI.Status()
        size = 1000
        self.data_array = [
            bytearray(('a'*size).encode("ascii"))
            for i in range(self.mpi.size-1)
        ]
        self.buf = [
            [self.data_array[i], size, MPI.CHAR]
            for i in range(self.mpi.size-1)
        ]
        tag = 1
        self.req_recv = [
            self.mpi.comm.Irecv(self.buf[i], source=i+1, tag=tag)
            for i in range(mpi.size-1)
        ]

    def init_send_individuals(self, pop):
        """ Send initial individuals """
        for world_rank in range(1, self.mpi.size):
            buffer = str(
                pop.individuals[pop.individuals_simulated]
            ).encode("ascii")
            print("Sending: {}".format(buffer))
            self.mpi.comm.Send(buffer, dest=world_rank, tag=1)
            pop.consume()

    def check_receive(self, pop):
        """ Check communication reception """
        tag = 1
        for i in range(self.mpi.size-1):
            msg = ""
            test = self.req_recv[i].Get_status(self.status)
            count = self.status.Get_count(MPI.CHAR)  # MPI.CHAR
            print("  count: {}\n  status: {}".format(
                count,
                self.status
            ))
            print("  Source: {}".format(self.status.Get_source()))
            if test and count:
                msg = self.data_array[i][:count].decode()
                print("Evolver: Message received:\n    {}".format(
                    msg.replace("\n", "\n    ")
                ))
                self.req_recv[i].Free()
            if msg and count:
                pop.simulation_complete()
                # print("Evolver from {}: {} ({})".format(i+1, msg, a))
                if pop.individuals_left:
                    # req_recv[i] = comm.Irecv(buf[i], source=i+1, tag=1)
                    self.req_recv[i] = self.mpi.comm.Irecv(
                        self.buf[i], source=i+1, tag=tag
                    )
                    buffer = str(
                        pop.individuals[pop.individuals_simulated]
                    ).encode("ascii")
                    print("Evolver: Sending back {}".format(buffer))
                    self.mpi.comm.Send(buffer, dest=i+1, tag=1)
                    print("Evolver: Message sent")
                    pop.consume()


class Communication:
    """ Communication """

    def __init__(self, mpi):
        super(Communication, self).__init__()
        self.mpi = mpi
        tag = 1
        self.req_recv = [
            self.mpi.comm.irecv(source=i+1, tag=tag)
            for i in range(mpi.size-1)
        ]

    def init_send_individuals(self, pop):
        """ Send initial individuals """
        for world_rank in range(1, self.mpi.size):
            buffer = pop.individuals[pop.individuals_simulated]
            print("Sending: {}".format(buffer))
            self.mpi.comm.send(
                buffer,
                dest=world_rank,
                tag=1
            )
            pop.consume()

    def check_receive(self, pop):
        """ Check communication reception """
        tag = 1
        for i in range(self.mpi.size-1):
            _, msg = self.req_recv[i].test()
            if msg:
                print(pop)
                individual = pop.individuals[pop.individuals_simulated-1]
                fitness = compute_fitness(
                    ".gazebo/models/"+individual,
                    "link_body_0"
                )
                print("Fitness for {}: {}".format(individual, fitness))
                pop.simulation_complete()
                # print("Evolver from {}: {} ({})".format(i+1, msg, a))
                if pop.individuals_left:
                    # req_recv[i] = comm.Irecv(buf[i], source=i+1, tag=1)
                    self.req_recv[i] = self.mpi.comm.irecv(
                        source=i+1,
                        tag=tag
                    )
                    buffer = pop.individuals[pop.individuals_simulated]
                    print("Evolver: Sending back {}".format(buffer))
                    self.mpi.comm.send(buffer, dest=i+1, tag=1)
                    print("Evolver: Message sent")
                    pop.consume()


def evolver():
    """ Writer """
    # Communication parameters
    mpi = MPIsettings()
    # Evolutions parameters
    pop = Population()
    for freq in np.linspace(0, 2, 5):
        name = "salamander_{}".format(float(freq)).replace(".", "d")
        print("Generating {} with frequency {}".format(name, freq))
        generate_walking(name, freq)
        pop.add_individuals([name])
    # Run evolution
    print("Evolver is running (rank={}, size={})".format(mpi.rank, mpi.size))
    assert mpi.rank == 0, "Rank of evolver must be 0, but is {}".format(mpi.rank)
    # Messaging
    comm = Communication(mpi)
    comm.init_send_individuals(pop)
    print("Evolver: All data has been sent, now waiting")
    while pop.individuals_left + pop.individuals_simulating:
        print(
            "Evolver: Individuals_left: {}, individuals_simulating: {}".format(
                pop.individuals_left,
                pop.individuals_simulating
            )
        )
        time.sleep(1e0)
        comm.check_receive(pop)
    print("Evolver: Closing")
    for i in range(mpi.size-1):
        mpi.comm.send("close", dest=i+1, tag=0)


if __name__ == '__main__':
    evolver()
