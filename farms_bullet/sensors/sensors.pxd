"""Cython sensors"""

from farms_amphibious.data.array cimport DoubleArray2D, DoubleArray3D


cdef class Sensors(dict):
    """Sensors"""
    pass


cdef class ContactsSensors(DoubleArray3D):
    """Model sensors"""
    cdef public unsigned int [:] animat_ids
    cdef public int [:] animat_links
    cdef public double inewtons
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


cdef class LinksStatesSensor(DoubleArray3D):
    """Links states sensor"""

    cdef public int animat
    cdef public object links
    cdef public object units
    cpdef public void collect(self, unsigned int iteration, object links)
