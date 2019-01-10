""" Plot """

import os
from salamander_msgs.salamander_kinematics_pb2 import ModelKinematics
from salamander_msgs.salamander_control_pb2 import SalamanderControl
import numpy as np
import matplotlib.pyplot as plt


def position(pos):
    """ Return array from position """
    return np.array([pos.x, pos.y, pos.z])


def extract_logs(path, log_type=ModelKinematics, log_file=None):
    """ Extract_logs """
    if log_file is None:
        log_file = "links_kinematics.pbdat"
    filepath = (
        os.path.expanduser('~')
        +"/"+path
        +"/logs/{}".format(log_file)
    )
    print("Opening file {}".format(filepath))
    with open(filepath, mode="rb") as _log_file:
        logs = log_type()
        print("Loading from string")
        logs.ParseFromString(_log_file.read())
    return logs


def positions(path, link_name):
    """ Link positions """
    print("Processing model {}".format(path))
    kin = extract_logs(path)
    print("Loaded kin")
    link = None
    for link in kin.links:
        if link.name == link_name:
            break
    print("Link name: {}".format(link.name))
    times = [state.time.sec+1e-9*state.time.nsec for state in link.state]
    print("  First 5 times: {}".format(times[:5]))
    print("  Last 5 times: {}".format(times[-5:]))
    pos = np.array([position(state.pose.position) for state in link.state])
    return pos


def joint_positions(path, joint_name):
    """ Joint positions """
    print("Processing model {}".format(path))
    kin = extract_logs(path)
    print("Loaded kin")
    joint = None
    for joint in kin.joints:
        if joint.name == joint_name:
            break
    print("Joint name: {}".format(joint.name))
    times = [state.time.sec+1e-9*state.time.nsec for state in joint.state]
    pos = np.array([state.position for state in joint.state])
    return pos, times


def plot_links_positions(path=".gazebo/models/salamander_new", figure=None):
    """ Plot position """
    if figure:
        plt.figure(figure)
    for i in range(12):
        link_name = "link_body_{}".format(i)
        pos = positions(path, link_name)
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
            pos = positions(path+model_name, link_name)
            plt.plot(pos[:, 0], pos[:, 1], label=model_name)
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()


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
    plt.show()


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
    plt.show()
