""" Writer """

import time
from mpi4py import MPI


def evolver():
    """ Writer """
    # Communication parameters
    comm = MPI.COMM_WORLD
    size = comm.Get_size()  # new: gives number of ranks in comm
    rank = comm.Get_rank()
    # Evolutions parameters
    # n_generations = 1
    n_population = 5
    individuals_left = n_population
    individuals_simulating = 0
    individuals_simulated = 0
    indivudual_types = [
        "biorob_salamander_walking",
        "biorob_salamander_swimming",
        "biorob_salamander_walking",
        "biorob_salamander_swimming",
        "biorob_salamander_walking"
    ]
    # Run evolution
    print("Evolver is running (rank={}, size={})".format(rank, size))
    assert rank == 0, "Rank of evolver must be 0, but is {}".format(rank)
    v = [bytearray(('a'*1000).encode("ascii")) for i in range(size-1)]
    # v = [array('u', '#') * 1000 for i in range(size-1)]
    print("Buffer: {}".format(v))
    buf = [[v[i], 1000, MPI.CHAR] for i in range(size-1)]
    tag = 1
    req_recv = [comm.Irecv(buf[i], source=i+1, tag=tag) for i in range(size-1)]
    for i, r in enumerate(range(1, size)):
        # buffer = [
        #     str({
        #         'message': "Message to process {}".format(i+1),
        #         "source": rank,
        #         'dest': i+1
        #     }).encode("ascii"),
        #     0,
        #     MPI.BYTE
        # ]
        buffer = str(indivudual_types[individuals_simulated]).encode("ascii")
        print("Sending: {}".format(buffer))
        comm.Send(buffer, dest=r, tag=1)
        individuals_left -= 1
        individuals_simulating += 1
        individuals_simulated += 1
    print("Evolver: All data has been sent, now waiting")
    # buf = [None, 0, MPI.BYTE]
    # req_recv = [comm.irecv(dest=i+1, tag=1) for i in range(size-1)]
    # Loop
    status = MPI.Status()
    # MPI.Request.Waitall([req_recv[0]], [status])
    # from IPython import embed; embed()
    while individuals_left + individuals_simulating:
        print(
            "Evolver: Individuals_left: {}, individuals_simulating: {}".format(
                individuals_left, individuals_simulating
            )
        )
        time.sleep(1e0)
        for i in range(size-1):
            msg = ""
            # a, msg = req_recv[i].test()
            # print("a: {}\nmsg: {}".format(a, msg))
            # test = req_recv[i].Test(status)
            test = req_recv[i].Get_status(status)
            # if test:
            #     re = MPI.Request.Waitall([req_recv[i]], [status])
            #     # re = MPI.Request.Wait(req_recv[i])
            #     print(re)
            # # test_status = req_recv[i].Get_status(status=status)
            # # count = bytearray(8)
            count = status.Get_count(MPI.CHAR)  # MPI.CHAR
            print("  count: {}\n  status: {}".format(
                count,
                status
            ))
            print("  Source: {}".format(status.Get_source()))
            if test and count:
                msg = v[i][:count].decode()
                print("Evolver: Message received:\n    {}".format(
                    msg.replace("\n", "\n    ")
                ))
                req_recv[i].Free()
            if msg and count:
                individuals_simulating -= 1
                # print("Evolver from {}: {} ({})".format(i+1, msg, a))
                if individuals_left:
                    # req_recv[i] = comm.Irecv(buf[i], source=i+1, tag=1)
                    req_recv[i] = comm.Irecv(buf[i], source=i+1, tag=tag)
                    buffer = str(
                        indivudual_types[individuals_simulated]
                    ).encode("ascii")
                    print("Evolver: Sending back {}".format(buffer))
                    comm.Send(buffer, dest=i+1, tag=1)
                    print("Evolver: Message sent")
                    individuals_left -= 1
                    individuals_simulating += 1
                    individuals_simulated += 1
                    # print(
                    #     "Evolver: New data sent back to process {}".format(
                    #         i+1
                    #     )
                    # )
    print("Evolver: Closing")
    for i in range(size-1):
        comm.Send("close".encode("ascii"), dest=i+1, tag=0)
    return


if __name__ == '__main__':
    evolver()
