"""Parse command line arguments"""

import argparse


def parse_args():
    """ Parse arguments """
    parser = argparse.ArgumentParser(description='Salamander simulation')
    parser.add_argument(
        '--timestep',
        type=float,
        dest='timestep',
        default=1e-3,
        help='Simulation timestep'
    )
    parser.add_argument(
        '--duration',
        type=float,
        dest='duration',
        default=100,
        help='Simulation duration'
    )
    parser.add_argument(
        '--n_solver_iters',
        type=int,
        dest='n_solver_iters',
        default=50,
        help='Number of solver iterations for physics simulation'
    )
    parser.add_argument(
        '-f', '--free_camera',
        action='store_true',
        dest='free_camera',
        default=False,
        help='Allow for free camera (User controlled)'
    )
    parser.add_argument(
        '-r', '--rotating_camera',
        action='store_true',
        dest='rotating_camera',
        default=False,
        help='Enable rotating camera'
    )
    parser.add_argument(
        '-t', '--top_camera',
        action='store_true',
        dest='top_camera',
        default=False,
        help='Enable top view camera'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        dest='fast',
        default=False,
        help='Remove real-time limiter'
    )
    parser.add_argument(
        '--record',
        action='store_true',
        dest='record',
        default=False,
        help='Record video'
    )
    parser.add_argument(
        '--fps',
        type=int,
        dest='fps',
        default=30,
        help='Video recording frames per second'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        dest='headless',
        default=False,
        help='Headless mode instead of using GUI'
    )
    # return parser.parse_args()
    args, _ = parser.parse_known_args()
    return args
