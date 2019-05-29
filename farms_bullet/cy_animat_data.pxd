# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=True
# cython: profile=True

"""Animat data"""

cdef class NetworkArray:
    pass


cdef class NetworkArray2D(NetworkArray):
    """Network array"""
    cdef public double[:, :] array
    cdef public unsigned int[2] size


cdef class NetworkArray3D(NetworkArray):
    """Network array"""
    cdef public double[:, :, :] array
    cdef public unsigned int[3] size


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


cdef class SensorArray(NetworkArray3D):
    """Sensor array"""
    cdef public unsigned int _n_iterations


cdef class JointsArray(NetworkArray2D):
    """Oscillator array"""
    pass


cdef class NetworkParameters:
    """Network parameter"""
    cdef public OscillatorArray oscillators
    cdef public ConnectivityArray connectivity
    cdef public JointsArray joints
    cdef public SensorArray contacts
    cdef public ConnectivityArray contacts_connectivity
    cdef public unsigned int iteration
