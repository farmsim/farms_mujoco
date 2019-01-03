#!/usr/bin/env python3
""" Test salamander run """

import argparse
# import threading
import multiprocessing
import salamander_pyrun as sr


def run(model_name):
    """ Run simulation from command line """
    world_path = "/.gazebo/models/{}/world.world".format(model_name)
    sr.run_simulation(world_path)


def parse_args():
    """ Parse arguments """
    parser = argparse.ArgumentParser(description='Model names')
    parser.add_argument(
        'model_names',
        type=str,
        nargs='+',
        help='Name of Gazebo model to simulate'
    )
    args = parser.parse_args()
    print(args.model_names)
    return args.model_names


def main():
    """ Main """
    model_names = parse_args()
    processes = [
        multiprocessing.Process(target=run, args=(model_name, ))
        for model_name in model_names
    ]
    for _p in processes:
        _p.start()
    for _p in processes:
        _p.join()


if __name__ == '__main__':
    main()
