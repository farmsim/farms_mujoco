""" Plot """

import os
from salamander_msgs.log_kinematics_pb2 import ModelKinematics
import numpy as np


def position(pos):
    """ Return array from position """
    return np.array([pos.x, pos.y, pos.z])


def positions(path, link_name):
    """ Main """
    print("Opening file")
    with open(
            os.path.expanduser('~')
            +"/"+path
            +"/logs/links_kinematics.pbdat",
            mode="rb"
    ) as log_file:
        kin = ModelKinematics()
        print("Loading from string")
        kin.ParseFromString(log_file.read())
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

# def positions(path, link):
#     """ Main """
#     print("Opening file")
#     with open(
#             os.path.expanduser('~')
#             +"/"+path
#             +"/logs/links_kinematics.pbdat",
#             mode="rb"
#     ) as log_file:
#         kin = msgs.LinksKinematics()
#         print("Loading from string")
#         kin.ParseFromString(log_file.read())
#     print("Loaded kin")
#     link = kin.links[0]
#     print("Link name: {}".format(link.name))
#     times = [k.sec+1e-9*k.nsec for k in link.link_kinematics]
#     print("  First 5 times: {}".format(times[:5]))
#     print("  Last 5 times: {}".format(times[-5:]))
#     pos = np.array([position(k.pos) for k in link.link_kinematics])
#     return pos


def compute_fitness(path, link):
    """ Main """
    pos = positions(path, link)
    distances = [np.linalg.norm(_pos) for _pos in pos[:, :2]]
    return max(distances)
