"""Animat"""

from ..simulations.element import SimulationElement


class Animat(SimulationElement):
    """Animat"""

    def __init__(self, options):
        super(Animat, self).__init__()
        self.options = options
