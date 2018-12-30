""" Run Gazebo island """

import os
import subprocess


def run_simulation(world_path="/.gazebo/models/salamander_new/world.world"):
    """ Run island """
    exe = "gzserver"
    verbose = "--verbose"
    # os.environ["GAZEBO_MASTER_URI"] = "localhost:11345"
    cmd = "{} {} {}".format(
        exe,
        verbose,
        os.environ["HOME"]+world_path
    )
    print(cmd)
    subprocess.call(cmd, shell=True)
    print("Simulation complete")


if __name__ == "__main__":
    run_island()
