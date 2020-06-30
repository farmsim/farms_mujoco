"""Parse command line arguments"""

import argparse


def parse_args():
    """ Parse arguments """
    parser = argparse.ArgumentParser(description='Salamander simulation')

    # Simulation
    parser.add_argument(
        '--timestep',
        type=float,
        dest='timestep',
        default=1e-3,
        help='Simulation timestep',
    )
    parser.add_argument(
        '--duration',
        type=float,
        dest='duration',
        default=1,
        help='Simulation duration',
    )
    parser.add_argument(
        '--pause',
        action='store_true',
        dest='pause',
        default=False,
        help='Pause simulation at start',
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        dest='fast',
        default=False,
        help='Remove real-time limiter',
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        dest='headless',
        default=False,
        help='Headless mode instead of using GUI',
    )

    # Camera
    parser.add_argument(
        '-f', '--free_camera',
        action='store_true',
        dest='free_camera',
        default=False,
        help='Allow for free camera (User controlled)',
    )
    parser.add_argument(
        '-r', '--rotating_camera',
        action='store_true',
        dest='rotating_camera',
        default=False,
        help='Enable rotating camera',
    )
    parser.add_argument(
        '-t', '--top_camera',
        action='store_true',
        dest='top_camera',
        default=False,
        help='Enable top view camera',
    )

    # Video recording
    parser.add_argument(
        '--record',
        action='store_true',
        dest='record',
        default=False,
        help='Record video',
    )
    parser.add_argument(
        '--fps',
        type=int,
        dest='fps',
        default=30,
        help='Video recording frames per second',
    )
    parser.add_argument(
        '--pitch',
        type=float,
        dest='video_pitch',
        default=-30,
        help='Camera pitch',
    )
    parser.add_argument(
        '--yaw',
        type=float,
        dest='video_yaw',
        default=0,
        help='Camera yaw',
    )
    parser.add_argument(
        '--distance',
        type=float,
        dest='video_distance',
        default=1,
        help='Camera distance',
    )

    # Pybullet
    parser.add_argument(
        '--n_solver_iters',
        type=int,
        dest='n_solver_iters',
        default=50,
        help='Number of solver iterations for physics simulation',
    )
    parser.add_argument(
        '--opengl2',
        action='store_true',
        dest='opengl2',
        default=False,
        help='Run simulation with OpenGL 2 instead of 3',
    )
    parser.add_argument(
        '--erp',
        type=float,
        dest='erp',
        default=1e-2,
        help='Pybullet ERP',
    )
    parser.add_argument(
        '--contact_erp',
        type=float,
        dest='contact_erp',
        default=0,
        help='Pybullet contact ERP',
    )
    parser.add_argument(
        '--friction_erp',
        type=float,
        dest='friction_erp',
        default=0,
        help='Pybullet friction ERP',
    )
    parser.add_argument(
        '--num_sub_steps',
        type=int,
        dest='num_sub_steps',
        default=0,
        help='Pybullet number of sub-steps',
    )
    parser.add_argument(
        '--max_num_cmd_per_1ms',
        type=int,
        dest='max_num_cmd_per_1ms',
        default=int(1e9),
        help='Pybullet maximum number of commands per millisecond',
    )
    parser.add_argument(
        '--residual_threshold',
        type=float,
        dest='residual_threshold',
        default=0,
        help='Pybullet solver residual threshold',
    )

    # return parser.parse_args()
    args, _ = parser.parse_known_args()
    return args
