"""Cython sensors"""

include '../data/convention.pxd'
from ..data.array cimport DoubleArray3D


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


cdef class JointsStatesSensor(DoubleArray3D):
    """Joint state sensor"""
    cdef public unsigned int model_id
    cdef public list joints_map
    cdef public double seconds
    cdef public double inewtons
    cdef public double itorques
    cpdef public tuple get_joints_states(self)
    cpdef public void update(self, unsigned int iteration)


cdef class LinksStatesSensor(DoubleArray3D):
    """Links states sensor"""

    cdef public int animat
    cdef public object links
    cdef public object units
    cpdef public object get_base_link_state(self)
    cpdef public object get_children_links_states(self)
    cpdef public void collect(self, unsigned int iteration, object links)
