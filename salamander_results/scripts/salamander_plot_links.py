#!/usr/bin/env python3
""" Plot positions """

import argparse
from salamander_results.plot import plot_links_positions
import matplotlib.pyplot as plt


def parse_args():
    """ Parse arguments """
    parser = argparse.ArgumentParser(
        description='Test parameters sweeping with Gazebo simulations'
    )
    parser.add_argument(
        "-p", '--path',
        type=str,
        dest='path',
        default=".gazebo/models/salamander_new",
        help='Path to model gazebo folder'
    )
    args = parser.parse_args()
    return args.path


def main():
    """ Main """
    path = parse_args()
    plot_links_positions(path)
    plt.show()


if __name__ == '__main__':
    main()
