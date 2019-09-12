"""Animat data"""

from .array cimport NetworkArray2D, NetworkArray3D


cdef class AnimatDataCy:
    """Network parameter"""
    cdef public OscillatorNetworkStateCy state
    cdef public NetworkParametersCy network
    cdef public JointsArrayCy joints
    cdef public SensorsDataCy sensors
    cdef public unsigned int iteration


cdef class NetworkParametersCy:
    """Network parameter"""
    cdef public OscillatorArrayCy oscillators
    cdef public ConnectivityArrayCy connectivity
    cdef public ConnectivityArrayCy contacts_connectivity
    cdef public ConnectivityArrayCy hydro_connectivity


cdef class OscillatorNetworkStateCy(NetworkArray3D):
    """Network state"""
    cdef public unsigned int n_oscillators
    cdef public unsigned int _iterations


cdef class OscillatorArrayCy(NetworkArray2D):
    """Oscillator array"""
    pass


cdef class ConnectivityArrayCy(NetworkArray2D):
    """Connectivity array"""
    pass


cdef class JointsArrayCy(NetworkArray2D):
    """Oscillator array"""
    pass


cdef class SensorsDataCy:
    """SensorsData"""
    cdef public ContactsArrayCy contacts
    cdef public ProprioceptionArrayCy proprioception
    cdef public GpsArrayCy gps
    cdef public HydrodynamicsArrayCy hydrodynamics


cdef class ContactsArrayCy(NetworkArray3D):
    """Sensor array"""

    cpdef double[:] reaction(self, unsigned int iteration, unsigned int sensor_i)
    cpdef double[:, :] reaction_all(self, unsigned int sensor_i)
    cpdef double[:] friction(self, unsigned int iteration, unsigned int sensor_i)
    cpdef double[:, :] friction_all(self, unsigned int sensor_i)
    cpdef double[:] total(self, unsigned int iteration, unsigned int sensor_i)
    cpdef double[:, :] total_all(self, unsigned int sensor_i)


cdef class ProprioceptionArrayCy(NetworkArray3D):
    """Proprioception array"""

    cpdef double position(self, unsigned int iteration, unsigned int joint_i)
    cpdef double[:] positions(self, unsigned int iteration)
    cpdef double[:, :] positions_all(self)
    cpdef double velocity(self, unsigned int iteration, unsigned int joint_i)
    cpdef double[:] velocities(self, unsigned int iteration)
    cpdef double[:, :] velocities_all(self)
    cpdef double[:] force(self, unsigned int iteration, unsigned int joint_i)
    cpdef double[:, :, :] forces_all(self)
    cpdef double[:] torque(self, unsigned int iteration, unsigned int joint_i)
    cpdef double[:, :, :] torques_all(self)
    cpdef double motor_torque(self, unsigned int iteration, unsigned int joint_i)
    cpdef double[:, :] motor_torques(self)


cdef class GpsArrayCy(NetworkArray3D):
    """Gps array"""

    cpdef public double[:] com_position(self, unsigned int iteration, unsigned int link_i)
    cpdef public double[:] com_orientation(self, unsigned int iteration, unsigned int link_i)
    cpdef public double[:] urdf_position(self, unsigned int iteration, unsigned int link_i)
    cpdef public double[:, :, :] urdf_positions(self)
    cpdef public double[:] urdf_orientation(self, unsigned int iteration, unsigned int link_i)
    cpdef public double[:] com_lin_velocity(self, unsigned int iteration, unsigned int link_i)
    cpdef public double[:, :, :] com_lin_velocities(self)
    cpdef public double[:] com_ang_velocity(self, unsigned int iteration, unsigned int link_i)


cdef class HydrodynamicsArrayCy(NetworkArray3D):
    """Hydrodynamics array"""

    cpdef public double[:, :, :] forces(self)
    cpdef public double[:, :, :] torques(self)
