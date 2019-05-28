"""Network"""

import numpy as np

from ..cy_animat_data import NetworkArray


class ODE(dict):
    """ODE"""

    __getattr__ = dict.__getitem__

    def __init__(self, solver, function, gradient=None):
        super(ODE, self).__init__({
            "solver": solver,
            "function": function,
            "gradient": gradient
        })


class CyODESolver:
    """ODE solver"""

    def __init__(self, ode, state, timestep, parameters, **kwargs):
        super(CyODESolver, self).__init__()
        self.ode = ode
        self._state = state
        self._n_dim = np.shape(state)[2]
        self._timestep = timestep
        self._iteration = kwargs.pop("iteration", 0)
        self._parameters = parameters

    @property
    def current_state(self):
        """State"""
        return self._state[self._iteration, 0]

    @property
    def current_dstate(self):
        """State derivative"""
        return self._state[self._iteration, 1]

    @property
    def iteration(self):
        """Iteration"""
        return self._iteration

    def step(self):
        """Control step"""
        self.ode.solver(
            self.ode.function,
            self._timestep,
            self._state,
            self._n_dim,
            self._iteration,
            *self._parameters.solver,
            self._parameters.function
        )
        self._iteration += 1


class ODESolver(CyODESolver):
    """ODE solver over time"""

    def __init__(self, ode, state, timestep, **kwargs):
        super(ODESolver, self).__init__(ode, state, timestep, **kwargs)
        iterations = np.shape(state)[0]
        self._times = np.arange(0, timestep*iterations, timestep)
        assert len(self._times) == iterations

    @property
    def time(self):
        """Time"""
        return self._times[self._iteration]


class NetworkParameters(ODE):
    """Network parameter"""

    def __init__(
            self,
            oscillators,
            connectivity,
            joints,
            contacts,
            contacts_connectivity
    ):
        super(NetworkParameters, self).__init__(
            [NetworkArray(np.zeros([  # Runge-Kutta parameters
                7,
                2*oscillators.shape()[1] + 1*joints.shape()[1]
            ]))],
            [oscillators, connectivity, joints, contacts, contacts_connectivity]
        )

    @property
    def rk4(self):
        """Rung-Kutta parameters"""
        return self.solver[0]

    @property
    def oscillators(self):
        """Oscillators parameters"""
        return self.function[0]

    @property
    def connectivity(self):
        """Connectivity parameters"""
        return self.function[1]

    @property
    def joints(self):
        """Joints parameters"""
        return self.function[2]

    @property
    def contacts(self):
        """Contacts parameters"""
        return self.function[3]

    @property
    def contacts_connectivity(self):
        """Contacts parameters"""
        return self.function[4]

    def to_ode_parameters(self):
        """Convert 2 arrays"""
        return ODE(
            [parameter.array for parameter in self.solver],
            [parameter.array for parameter in self.function]
            + [self.oscillators.shape()[1]]
            + [self.connectivity.shape()[0]]
            + [self.joints.shape()[1]]
            + [self.contacts.shape()[1]]
            + [self.contacts_connectivity.shape()[0]]
            + [0]
        )


class OscillatorArray(NetworkArray):
    """Oscillator array"""

    def __init__(self, array):
        super(OscillatorArray, self).__init__(array)
        self._array = array
        self._original_amplitudes_desired = np.copy(array[2])

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


class ConnectivityArray(NetworkArray):
    """Connectivity array"""

    @classmethod
    def from_parameters(cls, connections, weights, desired_phases):
        """From each parameter"""
        return cls(np.stack([connections, weights, desired_phases], axis=1))

    @property
    def connections(self):
        """Connections"""
        return self.array[:, [0, 1]]

    @property
    def weights(self):
        """Weights"""
        return self.array[:, 2]

    @property
    def desired_phases(self):
        """Weights"""
        return self.array[:, 3]


class SensorArray(NetworkArray):
    """Sensor array"""

    def __init__(self, array):
        super(SensorArray, self).__init__(array)
        self._array = array
        shape = np.shape(array)
        self._n_iterations, self._shape = shape[0], shape[1:]

    # @classmethod
    # def from_parameters(cls, n_iterations, sensors):
    #     """From each parameter"""
    #     array = np.zeros((n_iterations,)+np.shape(sensors))
    #     array[0, :] = sensors
    #     return cls(array)

    # @classmethod
    # def from_parameters(
    #         cls,
    #         n_iterations,
    #         proprioception,
    #         contacts,
    #         hydrodynamics
    # ):
    #     """From each parameter"""
    #     sensors = np.concatenate([proprioception, contacts, hydrodynamics])
    #     array = np.zeros([n_iterations, len(sensors)])
    #     array[0, :] = sensors
    #     return cls(
    #         array  # ,
    #         # len(proprioception),
    #         # len(contacts),
    #         # len(hydrodynamics)
    #     )

class JointsArray(NetworkArray):
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
