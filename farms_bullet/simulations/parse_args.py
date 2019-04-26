"""Parse command line arguments"""

import argparse


def parse_args():
    """ Parse arguments """
    parser = argparse.ArgumentParser(description="Salamander simulation")
    parser.add_argument(
        "-f", "--free_camera",
        action="store_true",
        dest="free_camera",
        default=False,
        help="Allow for free camera (User controlled)"
    )
    parser.add_argument(
        "-r", "--rotating_camera",
        action="store_true",
        dest="rotating_camera",
        default=False,
        help="Enable rotating camera"
    )
    parser.add_argument(
        "-t", "--top_camera",
        action="store_true",
        dest="top_camera",
        default=False,
        help="Enable top view camera"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        dest="fast",
        default=False,
        help="Remove real-time limiter"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        dest="record",
        default=False,
        help="Record video"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        dest="headless",
        default=False,
        help="Headless mode instead of using GUI"
    )
    parser.add_argument(
        "--no_plot",
        action="store_false",
        dest="plot",
        default=True,
        help="Plot at end of experiment for results analysis"
    )
    parser.add_argument(
        "-l", "--log",
        type=str,
        dest="log_path",
        default=False,
        help="Log data to path at end of experiment for results analysis"
    )
    parser.add_argument(
        "--log_extension",
        type=str,
        dest="log_extension",
        default="npy",
        choices=["npy", "txt", "csv"],
        help="Log data to path at end of experiment for results analysis"
    )
    return parser.parse_args()
