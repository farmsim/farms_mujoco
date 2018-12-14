""" Generate plugins configuration files """

from .gen_controller import generate_controller
from .gen_swimming import generate_swimming
from .gen_logs import generate_logs


def generate_plugins(**kwargs):
    """ Generate plugins """
    generate_controller(**kwargs)
    if kwargs["gait"] == "swimming":
        generate_swimming()
    generate_logs()
