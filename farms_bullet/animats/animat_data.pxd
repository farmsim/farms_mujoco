"""Animat data"""

from .array cimport NetworkArray2D, NetworkArray3D


cdef class AnimatData:
    """Network parameter"""
    cdef public OscillatorNetworkState state
    cdef public NetworkParameters network
    cdef public JointsArray joints
    cdef public Sensors sensors
    cdef public unsigned int iteration


cdef class NetworkParameters:
    """Network parameter"""
    cdef public OscillatorArray oscillators
    cdef public ConnectivityArray connectivity
    cdef public ConnectivityArray contacts_connectivity


cdef class OscillatorNetworkState(NetworkArray3D):
    """Network state"""
    cdef public unsigned int n_oscillators
    cdef public unsigned int _iterations


cdef class OscillatorArray(NetworkArray2D):
    """Oscillator array"""
    pass


cdef class ConnectivityArray(NetworkArray2D):
    """Connectivity array"""
    pass


cdef class JointsArray(NetworkArray2D):
    """Oscillator array"""
    pass


cdef class Sensors:
    """Sensors"""
    cdef public ContactsArray contacts
    # cdef public HydrodynamicsArray hydrodynamics


cdef class ContactsArray(NetworkArray3D):
    """Sensor array"""
    cdef public unsigned int _n_iterations


cdef class HydrodynamicsArray(NetworkArray3D):
    """Hydrodynamcics array"""
    pass