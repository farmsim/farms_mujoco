"""Arrays"""

include 'types.pxd'


cdef class DoubleArray:
    """Network array"""
    cpdef public unsigned int size(self, unsigned int index)


cdef class DoubleArray1D(DoubleArray):
    """Network array"""
    cdef readonly DTYPEv1 array


cdef class DoubleArray2D(DoubleArray):
    """Network array"""
    cdef readonly DTYPEv2 array


cdef class DoubleArray3D(DoubleArray):
    """Network array"""
    cdef readonly DTYPEv3 array


cdef class IntegerArray2D(DoubleArray):
    """Network array"""
    cdef readonly UITYPEv2 array
