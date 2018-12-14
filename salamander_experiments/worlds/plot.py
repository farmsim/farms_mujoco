""" Plot """

import log_kinematics_pb2 as lk

import numpy as np
import matplotlib.pyplot as plt


def position(pos):
    """ Return array from position """
    return np.array([pos.x, pos.y, pos.z])


def main():
    """ Main """
    print("Opening file")
    with open("logs/links_kinematics.pbdat", "r") as fp:
        kin = lk.LinksKinematics()
        print("Loading from string")
        kin.ParseFromString(fp.read())
    print("Loaded")
    for link in kin.links:
        print("Link name: {}".format(link.name))
        times = [k.sec+1e-9*k.nsec for k in link.link_kinematics]
        print("  First 5 times: {}".format(times[:5]))
        print("  Last 5 times: {}".format(times[-5:]))
        pos = np.array([position(k.pos) for k in link.link_kinematics])
        plt.plot(pos[:, 0], pos[:, 1], label=link.name)
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()
    return


if __name__ == '__main__':
    main()
