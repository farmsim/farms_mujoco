# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=True
# cython: profile=True

"""Animat data"""

import numpy as np


cdef class NetworkArray:
    """Network array"""

    def shape(self):
        """Array shape"""
        return np.shape(self.array)

    def copy_array(self):
        """Copy array"""
        return np.copy(self.array)


cdef class NetworkArray2D(NetworkArray):
    """Network array"""

    cdef public double[:, :] array

    def __init__(self, array):
        super(NetworkArray, self).__init__()
        self.array = array


cdef class NetworkArray3D(NetworkArray):
    """Network array"""

    cdef public double[:, :, :] array

    def __init__(self, array):
        super(NetworkArray, self).__init__()
        self.array = array


class OscillatorNetworkState(NetworkArray3D):
    """Network state"""

    # cdef public unsigned int n_oscillators
    # cdef public unsigned int _iterations

    def __init__(self, state, n_oscillators, iteration=0):
        self.n_oscillators = n_oscillators
        self._iteration = iteration
        super(OscillatorNetworkState, self).__init__(state)

    @classmethod
    def from_solver(cls, solver, n_oscillators):
        """From solver"""
        return cls(solver.state, n_oscillators, solver.iteration)

    def phases(self, iteration):
        """Phases"""
        return self.array[iteration, 0, :self.n_oscillators]

    def amplitudes(self, iteration):
        """Amplitudes"""
        return self.array[iteration, 0, self.n_oscillators:]

    def dphases(self, iteration):
        """Phases derivative"""
        return self.array[iteration, 1, :self.n_oscillators]

    def damplitudes(self, iteration):
        """Amplitudes derivative"""
        return self.array[iteration, 1, self.n_oscillators:]
