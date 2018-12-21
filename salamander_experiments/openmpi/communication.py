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
        tag = 1
        self.req_recv = [
            self.mpi.comm.irecv(source=i+1, tag=tag)
            for i in range(mpi.size-1)
        ]

    def init_send_individuals(self, pop):
        """ Send initial individuals """
        for world_rank in range(1, self.mpi.size):
            buffer = pop[pop.individuals_simulated].name
            print("Sending: {}".format(buffer))
            self.mpi.comm.send(
                buffer,
                dest=world_rank,
                tag=1
            )
            pop[pop.individuals_simulated].status = "in_simulation"

    def check_receive(self, pop):
        """ Check communication reception """
        tag = 1
        for i in range(self.mpi.size-1):
            _, msg = self.req_recv[i].test()
            if msg:
                pop[pop.individuals_simulated].status = "complete"
                # print("Evolver from {}: {} ({})".format(i+1, msg, a))
                if pop.individuals_pending:
                    # req_recv[i] = comm.Irecv(buf[i], source=i+1, tag=1)
                    self.req_recv[i] = self.mpi.comm.irecv(
                        source=i+1,
                        tag=tag
                    )
                    buffer = pop[pop.individuals_simulated].name
                    print("Evolver: Sending back {}".format(buffer))
                    self.mpi.comm.send(buffer, dest=i+1, tag=1)
                    print("Evolver: Message sent")
                    pop[pop.individuals_simulated].status = "in_simulation"
