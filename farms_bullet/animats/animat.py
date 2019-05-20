"""Animat"""

import pybullet
from ..simulations.element import SimulationElement


class Animat(SimulationElement):
    """Animat"""

    def __init__(self, options):
        super(Animat, self).__init__()
        self.options = options

    def n_joints(self):
        """Get number of joints"""
        return pybullet.getNumJoints(self._identity)
