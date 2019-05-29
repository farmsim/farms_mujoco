# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=True
# cython: profile=True

"""Animat data"""

cdef class NetworkArray:
    pass

cdef class NetworkArray2D(NetworkArray):
    """Network array"""

    cdef public double[:, :] array
