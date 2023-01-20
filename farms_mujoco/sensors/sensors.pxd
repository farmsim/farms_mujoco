"""Cython sensors"""

include 'types.pxd'
include 'sensor_convention.pxd'
import numpy as np

cimport numpy as np
from farms_core.array.array_cy cimport DoubleArray3D
from farms_core.sensors.data_cy cimport ContactsArrayCy, MusclesArrayCy


cpdef cycontacts2data(
    object physics,
    unsigned int iteration,
    ContactsArrayCy data,
    dict geom2data,
    double meters,
    double newtons,
)

cpdef cymusclesensors2data(
    object physics,
    unsigned int iteration,
    MusclesArrayCy data,
    np.ndarray[int, ndim=2] musclesensor2data,
    double meters,
    double velocity,
    double newtons,
)
