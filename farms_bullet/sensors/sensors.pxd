"""Cython sensors"""

include 'sensor_convention.pxd'
from farms_data.sensors.array cimport DoubleArray3D


cdef class Sensors(dict):
    """Sensors"""
    pass


cdef class ContactsSensors(DoubleArray3D):
    """Model sensors"""
    cdef public unsigned int [:] model_ids
    cdef public int [:] model_links
    cdef public double imeters
    cdef public double inewtons
    cpdef tuple get_contacts(self, unsigned int model_id, int model_link)
    cpdef void update(self, unsigned int iteration)


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

    cdef public int model
    cdef public object links
    cdef public double imeters
    cdef public double ivelocity
    cdef public double seconds
    cpdef public tuple get_base_link_state(self)
    cpdef public tuple get_children_links_states(self)
    cpdef public void update(self, unsigned int iteration)
