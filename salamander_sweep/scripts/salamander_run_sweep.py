#!/usr/bin/env python3
""" Salamander sweep """

import os
import argparse
from mpi4py import MPI

from salamander_sweep import Sweep, run_island


def parse_args():
    """ Parse arguments """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        'models',
        type=str,
        nargs='+',
        help='Models to be simulated'
    )
    args = parser.parse_args()
    return args.models


def main():
    """ Main """
    print("Starting simulation sweep")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        models = parse_args()
        sweep = Sweep(models)
        sweep.run()
    else:
        port = 11345 + rank - 1
        os.environ["GAZEBO_MASTER_URI"] = "localhost:{}".format(port)
        run_island()
    print("Simulation sweep COMPLETE")


if __name__ == '__main__':
    main()
