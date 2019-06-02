"""Animat data"""

import numpy as np
cimport numpy as np


cdef class AnimatData:
    """Network parameter"""

    def __init__(self, state, network, joints, sensors):
        super(AnimatData, self).__init__()
        self.state = state
        self.network = network
        self.joints = joints
        self.sensors = sensors
        self.iteration = 0


cdef class NetworkParameters:
    """Network parameter"""

    def __init__(
            self,
            oscillators,
            connectivity,
            contacts_connectivity
    ):
        super(NetworkParameters, self).__init__()
        self.oscillators = oscillators
        self.connectivity = connectivity
        self.contacts_connectivity = contacts_connectivity


cdef class OscillatorNetworkState(NetworkArray3D):
    """Network state"""

    def __init__(self, state, n_oscillators, iteration=0):
        super(OscillatorNetworkState, self).__init__(state)
        self.n_oscillators = n_oscillators
        self._iteration = iteration

    @classmethod
    def from_solver(cls, solver, n_oscillators):
        """From solver"""
        return cls(solver.state, n_oscillators, solver.iteration)

    def phases(self, unsigned int iteration):
        """Phases"""
        return self.array[iteration, 0, :self.n_oscillators]

    def amplitudes(self, unsigned int iteration):
        """Amplitudes"""
        return self.array[iteration, 0, self.n_oscillators:]

    def dphases(self, unsigned int iteration):
        """Phases derivative"""
        return self.array[iteration, 1, :self.n_oscillators]

    def damplitudes(self, unsigned int iteration):
        """Amplitudes derivative"""
        return self.array[iteration, 1, self.n_oscillators:]


cdef class OscillatorArray(NetworkArray2D):
    """Oscillator array"""

    @classmethod
    def from_parameters(cls, freqs, rates, amplitudes):
        """From each parameter"""
        return cls(np.array([freqs, rates, amplitudes]))

    @property
    def freqs(self):
        """Frequencies"""
        return self.array[0]

    @freqs.setter
    def freqs(self, value):
        """Frequencies"""
        self.array[0, :] = value

    @property
    def amplitudes_rates(self):
        """Amplitudes rates"""
        return self.array[1]

    @property
    def amplitudes_desired(self):
        """Amplitudes desired"""
        return self.array[2]

    @amplitudes_desired.setter
    def amplitudes_desired(self, value):
        """Amplitudes desired"""
        self.array[2, :] = value


cdef class ConnectivityArray(NetworkArray2D):
    """Connectivity array"""

    @classmethod
    def from_parameters(cls, connections, weights, desired_phases):
        """From each parameter"""
        return cls(np.stack([connections, weights, desired_phases], axis=1))

    @property
    def connections(self):
        """Connections"""
        return self.array[:][0, 1]

    @property
    def weights(self):
        """Weights"""
        return self.array[:][2]

    @property
    def desired_phases(self):
        """Weights"""
        return self.array[:][3]


cdef class JointsArray(NetworkArray2D):
    """Oscillator array"""

    @classmethod
    def from_parameters(cls, offsets, rates):
        """From each parameter"""
        return cls(np.array([offsets, rates]))

    @property
    def offsets(self):
        """Joints angles offsets"""
        return self.array[0]

    @property
    def rates(self):
        """Joints angles offsets rates"""
        return self.array[1]

    def set_body_offset(self, value, n_body_joints=11):
        """Body offset"""
        self.array[0, :n_body_joints] = value


cdef class Sensors:
    """Sensors"""

    def __init__(
            self,
            ContactsArray contacts,
            ProprioceptionArray proprioception,
            GpsArray gps,
            HydrodynamicsArray hydrodynamics
    ):
        super(Sensors, self).__init__()
        self.contacts = contacts
        self.proprioception = proprioception
        self.gps = gps
        self.hydrodynamics = hydrodynamics


cdef class ContactsArray(NetworkArray3D):
    """Sensor array"""
    pass


cdef class ProprioceptionArray(NetworkArray3D):
    """Proprioception array"""
    pass


cdef class GpsArray(NetworkArray3D):
    """Gps array"""
    pass


cdef class HydrodynamicsArray(NetworkArray3D):
    """Hydrodynamics array"""
    pass
