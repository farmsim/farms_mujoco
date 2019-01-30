""" Plot """

import os
from salamander_msgs.salamander_kinematics_pb2 import ModelKinematics
from salamander_msgs.salamander_control_pb2 import SalamanderControl
import numpy as np
import matplotlib.pyplot as plt

from .extract import (
    extract_logs,
    extract_positions,
    joint_positions
)


def plot_links_positions(path=".gazebo/models/salamander_new", figure=None):
    """ Plot position """
    if figure:
        plt.figure(figure)
    for i in range(12):
        link_name = "link_body_{}".format(i)
        pos = extract_positions(path, link_name)
        plt.plot(pos[:, 0], pos[:, 1], label=link_name)
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def plot_models_positions(path=".gazebo/models/", figure=None):
    """ Plot position """
    if figure:
        plt.figure(figure)
    log_file = "/logs/links_kinematics.pbdat"
    for folder in os.listdir(os.path.expanduser("~")+"/"+path):
        if os.path.isfile(os.path.expanduser("~")+"/"+path+folder+log_file):
            model_name = folder
            link_name = "link_body_{}".format(0)
            pos = extract_positions(path+model_name, link_name)
            plt.plot(pos[:, 0], pos[:, 1], label=model_name)
    plt.legend()
    plt.axis("equal")
    plt.grid(True)


def plot_joints_positions(path=".gazebo/models/salamander_new", figure=None):
    """ Plot position """
    if figure:
        plt.figure(figure)
    for i in range(11):
        joint_name = "link_body_{}".format(i+1)
        pos, times = joint_positions(path, joint_name)
        plt.plot(times, pos, label=joint_name)
    plt.legend()
    plt.grid(True)


def plot_joints_cmd_pos(path=".gazebo/models/salamander_new", figure=None):
    """ Plot position """
    if figure:
        plt.figure(figure)
    control_logs = extract_logs(
        path,
        log_type=SalamanderControl,
        log_file="control.pbdat"
    )
    for joint in control_logs.joints:
        times = [
            control.time.sec+1e-9*control.time.nsec
            for control in joint.control
        ]
        pos = [control.commands.position for control in joint.control]
        plt.plot(times, pos, label=joint.name)
    plt.legend()
    plt.grid(True)


def plot_joints_cmd_vel(path=".gazebo/models/salamander_new", figure=None):
    """ Plot position """
    if figure:
        plt.figure(figure)
    control_logs = extract_logs(
        path,
        log_type=SalamanderControl,
        log_file="control.pbdat"
    )
    for joint in control_logs.joints:
        times = [
            control.time.sec+1e-9*control.time.nsec
            for control in joint.control
        ]
        pos = [control.commands.velocity for control in joint.control]
        plt.plot(times, pos, label=joint.name)
    plt.legend()
    plt.grid(True)


def plot_joints_cmd_torque(path=".gazebo/models/salamander_new", figure=None):
    """ Plot position """
    if figure:
        plt.figure(figure)
    control_logs = extract_logs(
        path,
        log_type=SalamanderControl,
        log_file="control.pbdat"
    )
    for joint in control_logs.joints:
        times = [
            control.time.sec+1e-9*control.time.nsec
            for control in joint.control
        ]
        pos = [control.torque for control in joint.control]
        plt.plot(times, pos, label=joint.name)
    plt.legend()
    plt.grid(True)


def plot_joints_cmd_consumption(path=".gazebo/models/salamander_new", figure=None):
    """ Plot position """
    if figure:
        plt.figure(figure)
    control_logs = extract_logs(
        path,
        log_type=SalamanderControl,
        log_file="control.pbdat"
    )
    for joint in control_logs.joints:
        times = [
            control.time.sec+1e-9*control.time.nsec
            for control in joint.control
        ]
        pos = [control.consumption for control in joint.control]
        plt.plot(times, pos, label=joint.name)
    plt.legend()
    plt.grid(True)
