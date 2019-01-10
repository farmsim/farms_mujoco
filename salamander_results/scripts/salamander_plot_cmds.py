#!/usr/bin/env python3
""" Plot joints positions commands """

import argparse
from salamander_results import (
    plot_joints_cmd_pos,
    plot_joints_cmd_vel,
    plot_joints_cmd_torque
)


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
    plot_joints_cmd_pos(path, figure="Positions commands")
    plot_joints_cmd_vel(path, figure="Velocity commands")
    plot_joints_cmd_torque(path, figure="Motor torques")


if __name__ == '__main__':
    main()
