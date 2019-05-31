"""Cython sensors"""

from ..animats.array cimport NetworkArray2D, NetworkArray3D


cdef class Sensors(dict):
    """Sensors"""
    pass


cdef class ContactsSensors(NetworkArray3D):
    """Model sensors"""
    cdef public unsigned int [:] animat_ids
    cdef public int [:] animat_links
    cdef public unsigned int n_sensors
    cdef public list _contacts
    cdef public void _set_contact_forces(
        self,
        unsigned int iteration,
        unsigned int sensor,
        double[:] contact
    )
    cdef public void _set_total_force(
        self,
        unsigned int iteration,
        unsigned int sensor
    )
