""" Generate swimming """

import os
from collections import OrderedDict
from .yaml_utils import ordered_dump


def swim_parameters():  # gait="walking", frequency=1
    """ Network parameters """
    data = OrderedDict()
    data["swimming"] = True
    return data


def generate_config(data, filename="config/swim.yaml", verbose=False):
    """ Generate config """
    yaml_data = ordered_dump(data)
    if verbose:
        print(yaml_data)
    _filename = os.path.join(os.path.dirname(__file__), filename)
    with open(_filename, "w+") as yaml_file:
        yaml_file.write(yaml_data)
    print("{} generation complete".format(filename))


def generate_swimming():  # gait="walking", frequency=1
    """ Generate swimming config """
    data = swim_parameters()  # gait=gait, frequency=frequency
    generate_config(data)


def main():
    """ Main """
    generate_swimming()


if __name__ == '__main__':
    main()
