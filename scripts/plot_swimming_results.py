"""Plot swimming results"""

import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    """Main"""
    fish_name = "penpoint_gunnel"
    version_name = "version1"

    # Kinematics data from fish
    kinematics_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "farms_bullet",
        "experiments",
        "fish",
        fish_name,
        version_name,
        "kinematics",
        "kinematics.csv"
    )
    kinematics = np.loadtxt(kinematics_file)
    pose_fish = kinematics[:, :3]
    # pose_fish *= 1e-3
    position = np.ones(3)
    position[:2] = pose_fish[0, :2]
    orientation = np.zeros(3)
    orientation[2] = pose_fish[0, 2] + np.pi
    velocity = np.zeros(3)
    n_sample = 5
    velocity[:2] = pose_fish[n_sample, :2] - pose_fish[0, :2]
    sampling_timestep = 1e-2
    velocity /= n_sample*sampling_timestep
    kinematics = kinematics[:, 3:]
    kinematics = ((kinematics + np.pi) % (2*np.pi)) - np.pi

    # Simulation results
    simulation_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fish_results",
        "gps.npy",
    )
    simulation = np.load(simulation_file)
    pose_sim = simulation[:, 0, :3]
    pose_sim[:, :2] += pose_fish[0, :2] - pose_sim[0, :2]

    # Plot kinematics
    plt.plot(pose_fish[:, 0], pose_fish[:, 1], label="Fish data")

    # Plot simulation results
    plt.plot(pose_sim[:, 0], pose_sim[:, 1], label="Simulation data")

    # Plot options
    plt.xlabel("X axis [m]")
    plt.ylabel("Y axis [m]")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
