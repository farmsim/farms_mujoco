""" OpenMPI communication """

from mpi4py import MPI

class MPIsettings(dict):
    """ MPI settings """

    def __init__(self):
        super(MPIsettings, self).__init__()
        self["comm"] = MPI.COMM_WORLD
        self["size"] = self.comm.Get_size()
        self["rank"] = self.comm.Get_rank()

    @property
    def comm(self):
        """ Comm """
        return self["comm"]

    @property
    def size(self):
        """ Size """
        return self["size"]

    @property
    def rank(self):
        """ Rank """
        return self["rank"]


class Communication:
    """ Communication """

    def __init__(self, mpi):
        super(Communication, self).__init__()
        self.mpi = mpi
        self.req_recv = [
            None
            for i in range(mpi.size-1)
        ]
        self.track = {}

    def init_send_individuals(self, pop):
        """ Send initial individuals """
        for island_rank in range(1, self.mpi.size):
            self.simulate_new_individual(pop, island_rank)

    def check_receive(self, pop):
        """ Check communication reception """
        for island_rank in range(1, self.mpi.size):
            msg = False
            if self.req_recv[island_rank-1] is not None:
                _, msg = self.req_recv[island_rank-1].test()
            if msg:
                pop[self.track[island_rank]].status = "complete"
                self.track[island_rank] = None
                self.simulate_new_individual(pop, island_rank)

    def simulate_new_individual(self, pop, island_rank):
        """ Simulate new individual from population """
        for i, individual in enumerate(pop):
            if individual.status == "pending":
                self.send_individual(individual, island_rank)
                self.track[island_rank] = i
                break

    def send_individual(self, individual, island_rank):
        """ Send to gazebo island given by rank """
        tag = 1
        if individual.status == "pending":
            self.req_recv[island_rank-1] = self.mpi.comm.irecv(
                source=island_rank,
                tag=tag
            )
            buffer = individual.name
            print("Sending: {}".format(buffer))
            self.mpi.comm.send(buffer, dest=island_rank, tag=1)
            self.track[island_rank] = individual.name
            individual.status = "in_simulation"
        else:
            raise Exception("{} is not pending".format(individual.name))
