""" Plot """

import os
from salamander_msgs.log_kinematics_pb2 import ModelKinematics
import numpy as np
import matplotlib.pyplot as plt


def position(pos):
    """ Return array from position """
    return np.array([pos.x, pos.y, pos.z])


def extract_logs(path):
    """ Extract_logs """
    with open(
            os.path.expanduser('~')
            +"/"+path
            +"/logs/links_kinematics.pbdat",
            mode="rb"
    ) as log_file:
        kin = ModelKinematics()
        print("Loading from string")
        kin.ParseFromString(log_file.read())
    return kin


def positions(path, link_name):
    """ Main """
    print("Opening file")
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
