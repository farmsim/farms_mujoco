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
    cdef public ConnectivityArray hydro_connectivity


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
    cdef public ProprioceptionArray proprioception
    cdef public GpsArray gps
    cdef public HydrodynamicsArray hydrodynamics


cdef class ContactsArray(NetworkArray3D):
    """Sensor array"""

    cpdef double[:] reaction(self, unsigned int iteration, unsigned int sensor_i)
    cpdef double[:] friction(self, unsigned int iteration, unsigned int sensor_i)
    cpdef double[:] total(self, unsigned int iteration, unsigned int sensor_i)


cdef class ProprioceptionArray(NetworkArray3D):
    """Proprioception array"""

    cpdef double position(self, unsigned int iteration, unsigned int joint_i)
    cpdef double velocity(self, unsigned int iteration, unsigned int joint_i)
    cpdef double[:] force(self, unsigned int iteration, unsigned int joint_i)
    cpdef double[:] torque(self, unsigned int iteration, unsigned int joint_i)
    cpdef double motor_torque(self, unsigned int iteration, unsigned int joint_i)


cdef class GpsArray(NetworkArray3D):
    """Gps array"""

    cpdef public double[:] com_position(self, unsigned int iteration, unsigned int link_i)
    cpdef public double[:] com_orientation(self, unsigned int iteration, unsigned int link_i)
    cpdef public double[:] urdf_position(self, unsigned int iteration, unsigned int link_i)
    cpdef public double[:] urdf_orientation(self, unsigned int iteration, unsigned int link_i)
    cpdef public double[:] com_lin_velocity(self, unsigned int iteration, unsigned int link_i)
    cpdef public double[:] com_ang_velocity(self, unsigned int iteration, unsigned int link_i)


cdef class HydrodynamicsArray(NetworkArray3D):
    """Hydrodynamics array"""
    pass
