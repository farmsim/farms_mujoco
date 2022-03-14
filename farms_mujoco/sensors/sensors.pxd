"""Cython sensors"""

include 'types.pxd'
include 'sensor_convention.pxd'
from farms_data.array.array_cy cimport DoubleArray3D
from farms_data.sensors.data_cy cimport ContactsArrayCy


cpdef cycontacts2data(
    object physics,
    unsigned int iteration,
    ContactsArrayCy data,
    dict geom2data,
    set geom_set,
    double meters,
    double newtons,
)
