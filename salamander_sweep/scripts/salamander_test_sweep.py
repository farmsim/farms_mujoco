#!/usr/bin/env python3
""" Test parameter sweep """

import argparse
from subprocess import call

from salamander_generation import generate_walking, ControlParameters
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
        "-f", '--n_frequencies',
        type=int,
        dest='n_frequencies',
        default=5,
        help='Number of frequencies to simulate [0-2]'
    )
    parser.add_argument(
        "-b", '--n_bias',
        type=int,
        dest='n_bias',
        default=1,
        help='Number of body bias to simulate [0-0.1]'
    )
    parser.add_argument(
        "-a", '--n_amplitude',
        type=int,
        dest='n_amplitude',
        default=1,
        help='Number of body amplitude to simulate [0-0.3]'
    )
    args = parser.parse_args()
    return args

def main():
    """ Main """
    names = []
    args = parse_args()
    for amplitude in np.linspace(0.3, 0, args.n_amplitude):
        for bias in np.linspace(0, 0.2, args.n_bias):
            for freq in np.linspace(0.5, 1.5, args.n_frequencies):
                name = "salamander_f{:.4f}_a{:.4f}_b{:.4f}".format(
                    float(freq),
                    float(amplitude),
                    float(bias)
                ).replace(".", "d")
                names.append(name)
                control_data = ControlParameters(
                    gait="walking",
                    frequency=float(freq),
                    body_amplitude=float(amplitude),
                    body_bias=float(bias)
                )
                generate_walking(
                    name=name,
                    control_parameters=control_data
                )
    cmd = "mpiexec -n {} salamander_run_sweep.py {}".format(
        args.n_processes,
        " ".join(names)
    )
    print(cmd)
    call(cmd, shell=True)


if __name__ == '__main__':
    main()
