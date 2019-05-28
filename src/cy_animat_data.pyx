"""Animat data"""

import numpy as np


class NetworkArray:
    """Network array"""

    def __init__(self, array):
        super(NetworkArray, self).__init__()
        self.array = array

    def shape(self):
        """Array shape"""
        return np.shape(self.array)

    def copy_array(self):
        """Copy array"""
        return np.copy(self.array)
