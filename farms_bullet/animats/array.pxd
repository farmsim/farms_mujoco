"""Arrays"""

cdef class NetworkArray:
    pass


cdef class NetworkArray2D(NetworkArray):
    """Network array"""
    cdef readonly double[:, :] array
    cdef readonly unsigned int[2] size


cdef class NetworkArray3D(NetworkArray):
    """Network array"""
    cdef readonly double[:, :, :] array
    cdef readonly unsigned int[3] size
