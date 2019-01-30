#!/usr/bin/env python3
""" Plot joints positions commands """

import argparse
import matplotlib.pyplot as plt
from salamander_results import extract_final_consumption
from salamander_results.plot import (
    plot_joints_cmd_pos,
    plot_joints_cmd_vel,
    plot_joints_cmd_torque,
    plot_joints_cmd_consumption
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
    plot_joints_cmd_consumption(path, figure="Torque consumption")
    consumption = extract_final_consumption(path)
    print("Consumption:{}".format("\n  ".join([
        "{}: {}".format(joint, consumption[joint])
        for joint in consumption
    ])))
    plt.show()


if __name__ == '__main__':
    main()
