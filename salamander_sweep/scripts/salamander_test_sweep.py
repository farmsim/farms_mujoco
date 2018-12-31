#!/usr/bin/env python3
""" Test parameter sweep """

import argparse
from subprocess import call

from salamander_generation import generate_walking, control_parameters
import numpy as np


def parse_args():
    """ Parse arguments """
    parser = argparse.ArgumentParser(
        description='Test parameters sweeping with Gazebo simulations'
    )
    parser.add_argument(
        "-p", '--n_processes',
        type=int,
        dest='n_processes',
        default=2,
        help='Number of processes to use'
    )
    parser.add_argument(
        "-i", '--n_individuals',
        type=int,
        dest='n_individuals',
        default=5,
        help='Number of individuals to simulate'
    )
    args = parser.parse_args()
    return args.n_processes, args.n_individuals


def main():
    """ Main """
    names = []
    n_processes, n_individuals = parse_args()
    for freq in np.linspace(0, 2, n_individuals):
        name = "salamander_{:.2f}".format(float(freq)).replace(".", "d")
        names.append(name)
        control_data = control_parameters(gait="walking", frequency=float(freq))
        generate_walking(
            name=name,
            control_plugin_parameters=control_data
        )
    cmd = "mpiexec -n {} salamander_run_sweep.py {}".format(
        n_processes,
        " ".join(names)
    )
    print(cmd)
    call(cmd, shell=True)


if __name__ == '__main__':
    main()
