""" Test evolver """

import time
from mpi4py import MPI


def test_evolver():
    """ Reader """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()  # new: gives number of ranks in comm
    rank = comm.Get_rank()
    print("EvolverTest running (rank={}, size={})".format(rank, size))
    req_recv = comm.irecv(dest=0, tag=1)
    req_recv_master = comm.irecv(dest=0, tag=0)
    while True:
        time.sleep(1e-1)
        # Receive message
        a, msg = req_recv.test()
        a_master, msg_master = req_recv_master.test()
        if msg_master == "close":
            print("Process {}: Close received".format(rank))
            break
        if msg:
            req_recv = comm.irecv(dest=0, tag=1)
            if False:
                print('Process {}: Message is {}'.format(rank, msg))
            print('Process {}: NOW COMPUTING'.format(rank))
            # Computing
            time.sleep(0.5)
            # Computation complete
            data = "Message received from {} and processed".format(rank)
            # print("Process {}: Ready to send".format(rank))
            comm.send(data, dest=0, tag=1)
            # print("Process {}: Data sent back".format(rank))
    print("Process {}: Closing".format(rank))
    return


if __name__ == '__main__':
    test_evolver()
