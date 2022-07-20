"""Cython sensors"""

include 'types.pxd'
include 'sensor_convention.pxd'
from farms_core.array.array_cy cimport DoubleArray3D
from farms_core.sensors.data_cy cimport ContactsArrayCy


cpdef cycontacts2data(
    object physics,
    unsigned int iteration,
    ContactsArrayCy data,
    dict geom2data,
    double meters,
    double newtons,
)
