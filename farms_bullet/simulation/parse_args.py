"""Parse command line arguments"""

import argparse


def argument_parser():
    """Argument parser"""
    parser = argparse.ArgumentParser(
        description='FARMS simulation with Pybullet',
        formatter_class=(
            lambda prog:
            argparse.HelpFormatter(prog, max_help_position=50)
        ),
    )

    # Simulation
    parser.add_argument(
        '--timestep',
        type=float,
        default=1e-3,
        help='Simulation timestep',
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=1,
        help='Simulation duration',
    )
    parser.add_argument(
        '--pause',
        action='store_true',
        default=False,
        help='Pause simulation at start',
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        default=False,
        help='Remove real-time limiter',
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        default=False,
        help='Headless mode instead of using GUI',
    )
    parser.add_argument(
        '--noprogress', '--npb',
        action='store_false',
        dest='show_progress',
        help='Hide progress bar',
    )

    # Units
    parser.add_argument(
        '--meters',
        type=float,
        default=1,
        help='Unit scaling of meters within physics engine',
    )
    parser.add_argument(
        '--seconds',
        type=float,
        default=1,
        help='Unit scaling of seconds within physics engine',
    )
    parser.add_argument(
        '--kilograms',
        type=float,
        default=1,
        help='Unit scaling of kilograms within physics engine',
    )

    # Camera
    parser.add_argument(
        '-f', '--free_camera',
        action='store_true',
        default=False,
        help='Allow for free camera (User controlled)',
    )
    parser.add_argument(
        '-r', '--rotating_camera',
        action='store_true',
        default=False,
        help='Enable rotating camera',
    )
    parser.add_argument(
        '-t', '--top_camera',
        action='store_true',
        default=False,
        help='Enable top view camera',
    )

    # Video recording
    parser.add_argument(
        '--record',
        action='store_true',
        default=False,
        help='Record video',
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Video recording frames per second',
    )
    parser.add_argument(
        '--video_pitch',
        type=float,
        default=-30,
        help='Camera pitch',
    )
    parser.add_argument(
        '--video_yaw',
        type=float,
        default=0,
        help='Camera yaw',
    )
    parser.add_argument(
        '--video_distance',
        type=float,
        default=1,
        help='Camera distance',
    )

    # Pybullet
    parser.add_argument(
        '--lcp',
        type=str,
        choices=('si', 'dantzig', 'pgs'),
        default='dantzig',
        help='Constraint solver LCP type',
    )
    parser.add_argument(
        '--n_solver_iters',
        type=int,
        default=50,
        help='Number of solver iterations for physics simulation',
    )
    parser.add_argument(
        '--gravity',
        nargs=3,
        type=float,
        metavar=('x', 'y', 'z'),
        default=(0, 0, -9.81),
        help='Gravity',
    )
    parser.add_argument(
        '--opengl2',
        action='store_true',
        default=False,
        help='Run simulation with OpenGL 2 instead of 3',
    )
    parser.add_argument(
        '--erp',
        type=float,
        default=0,
        help='Pybullet ERP',
    )
    parser.add_argument(
        '--contact_erp',
        type=float,
        default=0,
        help='Pybullet contact ERP',
    )
    parser.add_argument(
        '--friction_erp',
        type=float,
        default=0,
        help='Pybullet friction ERP',
    )
    parser.add_argument(
        '--num_sub_steps',
        type=int,
        default=0,
        help='Pybullet number of sub-steps',
    )
    parser.add_argument(
        '--max_num_cmd_per_1ms',
        type=int,
        default=int(1e9),
        help='Pybullet maximum number of commands per millisecond',
    )
    parser.add_argument(
        '--residual_threshold',
        type=float,
        default=1e-6,
        help='Pybullet solver residual threshold',
    )
    return parser


def parse_args():
    """Parse arguments"""
    parser = argument_parser()
    # return parser.parse_args()
    args, _ = parser.parse_known_args()
    return args
