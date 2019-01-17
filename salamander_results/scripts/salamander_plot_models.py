#!/usr/bin/env python3
""" Plot all positions """

import argparse
from salamander_results.plot import plot_models_positions


def parse_args():
    """ Parse arguments """
    parser = argparse.ArgumentParser(
        description='Test parameters sweeping with Gazebo simulations'
    )
    parser.add_argument(
        "-p", '--path',
        type=str,
        dest='path',
        default=".gazebo/models/",
        help='Path to gazeb models folder'
    )
    args = parser.parse_args()
    return args.path


def main():
    """ Main """
    path = parse_args()
    plot_models_positions(path)


if __name__ == '__main__':
    main()
