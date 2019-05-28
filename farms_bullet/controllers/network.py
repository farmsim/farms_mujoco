"""Network"""

import numpy as np

from ..cy_animat_data import NetworkArray2D, NetworkArray3D


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
            [NetworkArray2D(np.zeros([  # Runge-Kutta parameters
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
