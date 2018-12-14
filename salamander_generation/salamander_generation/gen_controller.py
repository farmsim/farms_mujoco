""" Generate controller """

import os
from collections import OrderedDict
import numpy as np

from .yaml_utils import ordered_dump


def control_parameters(gait="walking", frequency=1):
    """ Network parameters """
    print("Generating config for {} gait at {} [Hz]".format(gait, frequency))
    data = OrderedDict()
    joints = OrderedDict()
    data["joints"] = joints
    # Morphology
    n_body = 11
    n_legs = 2
    # joints["n_body"] = n_body
    # joints["n_legs"] = n_legs
    # Body
    bias = 0
    for i in range(n_body):
        amplitude = 0.3 if gait == "walking" else 0.1+i*0.4/n_body
        joint = OrderedDict()
        joints["link_body_{}".format(i+1)] = joint
        joint["type"] = "position"
        joint["amplitude"] = (
            float(amplitude*np.sin(2*np.pi*i/n_body))
            if gait == "walking"
            else float(amplitude)
        )
        joint["frequency"] = frequency
        joint["phase"] = 0 if gait == "walking" else float(2*np.pi*i/n_body)
        joint["bias"] = bias
        pids = OrderedDict()
        joint["pid"] = pids
        pid_pos = OrderedDict()
        pids["position"] = pid_pos
        pid_pos["p"] = 1e1 if gait == "walking" else 1e1
        pid_pos["i"] = 1e0 if gait == "walking" else 0
        pid_pos["d"] = 0
        pid_vel = OrderedDict()
        pids["velocity"] = pid_vel
        pid_vel["p"] = 1e-2 if gait == "walking" else 1e-2
        pid_vel["i"] = 1e-3 if gait == "walking" else 0
        pid_vel["d"] = 0
    for leg_i in range(n_legs):
        for side_i, side in enumerate(["L", "R"]):
            for part_i in range(3):
                joint = OrderedDict()
                joint_name = "link_leg_{}_{}_{}".format(
                    leg_i,
                    side,
                    part_i
                )
                joints[joint_name] = joint
                joint["type"] = "position"  # if gait=="walking" else "torque"
                joint["amplitude"] = (
                    float(0.6 if part_i == 0 else 0.1)
                    if gait == "walking"
                    else 0.0
                )
                joint["frequency"] = (
                    float(frequency)
                    if gait == "walking"
                    else 0
                )
                joint["phase"] = (
                    float(
                        np.pi*np.abs(leg_i-side_i)
                        + (0 if part_i == 0 else 0.5*np.pi)
                    )
                    if gait == "walking"
                    else 0
                )
                joint["bias"] = (
                    float(0 if part_i == 0 else 0.1)
                    if gait == "walking"
                    else -2*np.pi/5 if part_i == 0
                    else 0
                )
                pids = OrderedDict()
                joint["pid"] = pids
                pid_pos = OrderedDict()
                pids["position"] = pid_pos
                pid_pos["p"] = 1e1
                pid_pos["i"] = 1e0
                pid_pos["d"] = 0
                pid_vel = OrderedDict()
                pids["velocity"] = pid_vel
                pid_vel["p"] = 1e-3
                pid_vel["i"] = 1e-4
                pid_vel["d"] = 0
    return data


def generate_config(data, filename="config/control.yaml", verbose=False):
    """ Generate config """
    _filename = os.path.join(os.path.dirname(__file__), filename)
    yaml_data = ordered_dump(data)
    if verbose:
        print(yaml_data)
    with open(_filename, "w+") as yaml_file:
        yaml_file.write(yaml_data)
    print("{} generation complete".format(_filename))


def generate_controller(gait="walking", frequency=1):
    """ Generate controller config """
    data = control_parameters(gait=gait, frequency=frequency)
    generate_config(data)


def main():
    """ Main """
    generate_controller()


if __name__ == '__main__':
    main()
