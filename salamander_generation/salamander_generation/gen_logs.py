""" Generate swimming """

import os
from collections import OrderedDict
from .yaml_utils import ordered_dump


def logs_parameters():
    """ Network parameters """
    data = OrderedDict()
    data["config"] = (
        "/.gazebo/models/"
        + "biorob_salamander_swimming/"
        + "config/log_kinematics.yaml"
    )
    return data


def generate_yaml(data, filename="config/plugin.yaml", verbose=False):
    """ Generate yaml """
    _filename = os.path.join(os.path.dirname(__file__), filename)
    yaml_data = ordered_dump(data)
    if verbose:
        print(yaml_data)
    with open(_filename, "w+") as yaml_file:
        yaml_file.write(yaml_data)
    print("{} generation complete".format(_filename))


def logs_config():
    """ Generate logging config """
    data = {
        "filename": "logs/links_kinematics.pbdat",
        "links": {
            "link_body_0": {
                "frequency": 100
            },
            "link_body_1": {
                "frequency": 10
            }
        }
    }
    return data


def generate_logs():
    """ Generate logging config """
    data = logs_parameters()
    generate_yaml(data, filename="config/log_kinematics.yaml")
    data = logs_config()
    generate_yaml(data, filename="config/config_log_kinematics.yaml")


def main():
    """ Main """
    generate_logs()


if __name__ == '__main__':
    main()
