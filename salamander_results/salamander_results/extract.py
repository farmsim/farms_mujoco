""" Extract """

import os

import numpy as np

from salamander_msgs.salamander_kinematics_pb2 import ModelKinematics
from salamander_msgs.salamander_control_pb2 import SalamanderControl


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


def extract_positions(path, link_name):
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


def extract_consumption(path=".gazebo/models/salamander_new"):
    """ Consumption """
    control_logs = extract_logs(
        path,
        log_type=SalamanderControl,
        log_file="control.pbdat"
    )
    return {
        joint.name: [control.consumption for control in joint.control]
        for joint in control_logs.joints
    }


def extract_final_consumption(path=".gazebo/models/salamander_new"):
    """ Final consumption """
    consumption = extract_consumption(path)
    return {
        joint: consumption[joint][-1]
        for joint in consumption
    }
