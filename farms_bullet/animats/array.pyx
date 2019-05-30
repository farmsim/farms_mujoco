"""Arrays"""

import numpy as np
cimport numpy as np


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

    def __init__(self, array):
        super(NetworkArray, self).__init__()
        self.array = array
        shape = np.array(np.shape(array), dtype=np.uint)
        cdef unsigned int i
        for i in range(2):
            self.size[i] = shape[i]


cdef class NetworkArray3D(NetworkArray):
    """Network array"""

    def __init__(self, array):
        super(NetworkArray, self).__init__()
        self.array = array
        shape = np.array(np.shape(array), dtype=np.uint)
        cdef unsigned int i
        for i in range(3):
            self.size[i] = shape[i]
