""" Coomunication with cpp MPI """

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
