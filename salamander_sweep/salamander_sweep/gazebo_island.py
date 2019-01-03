""" Gazebo island """

import os
import subprocess
import time

from .communication import MPIsettings
from salamander_pyrun import run_simulation

def _run_island(world_path="/.gazebo/models/salamander_new/world.world"):
    """ Run island """
    exe = "gzserver"
    verbose = "--verbose"
    cmd = "{} {} {}".format(
        exe,
        verbose,
        os.environ["HOME"]+world_path
    )
    print(cmd)
    subprocess.call(cmd, shell=True)
    print("Simulation complete")


def run_island():
    """ Main """
    # Communication parameters
    mpi = MPIsettings()
    print("EvolverTest running (rank={}, size={})".format(mpi.rank, mpi.size))
    req_recv = mpi.comm.irecv(source=0, tag=1)
    req_recv_master = mpi.comm.irecv(source=0, tag=0)
    while True:
        time.sleep(1e-1)
        # Receive message
        _, msg = req_recv.test()
        _, msg_master = req_recv_master.test()
        if msg_master == "close":
            print("Process {}: Close received".format(mpi.rank))
            break
        if msg:
            req_recv = mpi.comm.irecv(source=0, tag=1)
            print('Process {}: NOW COMPUTING'.format(mpi.rank))
            # Computing
            # _run_island("/.gazebo/models/{}/world.world".format(msg))
            run_simulation("/.gazebo/models/{}/world.world".format(msg))
            # Computation complete
            data = "Message received from {} and processed".format(mpi.rank)
            # print("Process {}: Ready to send".format(rank))
            mpi.comm.send(data, dest=0, tag=1)
            # print("Process {}: Data sent back".format(rank))
        if msg_master:
            if msg_master == "close":
                break
    print("Process {}: Closing".format(mpi.rank))


if __name__ == '__main__':
    run_island()
