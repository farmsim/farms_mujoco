"""Arrays"""

cdef class NetworkArray:
    pass


cdef class NetworkArray2D(NetworkArray):
    """Network array"""
    cdef public double[:, :] array
    cdef public unsigned int[2] size


cdef class NetworkArray3D(NetworkArray):
    """Network array"""
    cdef public double[:, :, :] array
    cdef public unsigned int[3] size
