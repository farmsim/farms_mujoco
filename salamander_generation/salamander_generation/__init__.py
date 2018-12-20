""" Salamander generation package

This package is meant for generating salamander models for the Gazebo simulator.
"""

__all__ = ["generate_all", "test_entity", "generate_walking"]


from .gen_all import generate_all
from .gen_entity import test_entity
from .gen_entity import generate_walking
