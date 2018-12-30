#!/usr/bin/env python3
""" Test salamander run """

import argparse
import salamander_pyrun as sr


def run(model_name):
    """ Run simulation from command line """
    world_path = "/.gazebo/models/{}/world.world".format(model_name)
    sr.run_simulation(world_path)


def parse_args():
    """ Parse arguments """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        'model_name',
        type=str,
        nargs='+',
        help='Name of Gazebo model to simulate'
    )
    args = parser.parse_args()
    print(args.model_name)
    return args.model_name


def main():
    """ Main """
    model_names = parse_args()
    for model_name in model_names:
        run(model_name)


if __name__ == '__main__':
    main()
