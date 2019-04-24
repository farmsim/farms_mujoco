"""Network"""

import numpy as np
from ..cy_controller import ode_oscillators_sparse, ode_amplitude, rk4
from .convention import bodyjoint2index, legjoint2index
from .control_options import SalamanderControlOptions


class ODE(list):
    """ODE"""

    def __init__(self, solver, function):
        super(ODE, self).__init__([solver, function])

    @property
    def solver(self):
        """Solver"""
        return self[0]

    @property
    def function(self):
        """Function"""
        return self[1]


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
        # print(self._n_dim)
        # print(self._parameters.function[-1])
        # print([np.shape(parameter) for parameter in self._parameters.function])
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


class NetworkArray:
    """Network array"""

    def __init__(self, array):
        super(NetworkArray, self).__init__()
        self._array = array

    @property
    def array(self):
        """Array"""
        return self._array


class OscillatorNetworkState(NetworkArray):
    """Network state"""

    def __init__(self, state, n_oscillators, iteration=0):
        self.n_oscillators = n_oscillators
        self._iteration = iteration
        super(OscillatorNetworkState, self).__init__(state)

    @classmethod
    def from_initial_state(cls, initial_state, n_iterations, n_oscillators):
        """From initial state"""
        state = np.zeros(
            [n_iterations, 2, np.shape(initial_state)[0]],
            dtype=np.float64
        )
        state[0, 0] = np.array(initial_state)
        return cls(state, n_oscillators)

    @classmethod
    def from_solver(cls, solver, n_oscillators):
        """From solver"""
        return cls(solver.state, n_oscillators, solver.iteration)

    def phases(self, iteration):
        """Phases"""
        return self.array[iteration, 0, :self.n_oscillators]

    def amplitudes(self, iteration):
        """Amplitudes"""
        return self.array[iteration, 0, self.n_oscillators:]

    def dphases(self, iteration):
        """Phases derivative"""
        return self.array[iteration, 1, :self.n_oscillators]

    def damplitudes(self, iteration):
        """Amplitudes derivative"""
        return self.array[iteration, 1, self.n_oscillators:]


class SalamanderNetworkParameters(ODE):
    """Salamander network parameter"""

    def __init__(self, oscillators, connectivity, joints):
        super(SalamanderNetworkParameters, self).__init__(
            [NetworkArray(np.zeros([7, 2*np.shape(oscillators.array)[1]]))],
            [oscillators, connectivity, joints]
        )

    @classmethod
    def from_gait(cls, gait):
        """ Salamander network parameters from gait"""
        return (
            cls.for_swimming()
            if gait == "swimming"
            else cls.for_walking()
        )

    @classmethod
    def for_walking(cls):
        """Salamander swimming network"""
        oscillators = OscillatorArray.for_walking()
        connectivity = ConnectivityArray.for_walking()
        joints = JointsArray.for_walking()
        return cls(oscillators, connectivity, joints)

    @classmethod
    def for_swimming(cls):
        """Salamander swimming network"""
        oscillators = OscillatorArray.for_swimming()
        connectivity = ConnectivityArray.for_swimming()
        joints = JointsArray.for_swimming()
        return cls(oscillators, connectivity, joints)

    @property
    def rk4(self):
        """Rung-Kutta parameters"""
        return self[0][0]

    @property
    def oscillators(self):
        """Oscillators parameters"""
        return self[1][0]

    @property
    def connectivity(self):
        """Connectivity parameters"""
        return self[1][1]

    @property
    def joints(self):
        """Joints parameters"""
        return self[1][2]

    def to_ode_parameters(self):
        """Convert 2 arrays"""
        return ODE(
            [parameter.array for parameter in self.solver],
            [parameter.array for parameter in self.function]
            + [np.shape(self.oscillators.array)[1]]
            + [np.shape(self.connectivity.array)[0]]
        )


class OscillatorArray(NetworkArray):
    """Oscillator array"""

    @classmethod
    def from_parameters(cls, freqs, rates, amplitudes):
        """From each parameter"""
        return cls(np.array([freqs, rates, amplitudes]))

    @classmethod
    def for_walking(cls):
        """Parameters for walking"""
        n_body = 11
        n_dof_legs = 3
        n_legs = 4
        n_joints = n_body + n_legs*n_dof_legs
        n_oscillators = 2*(n_joints)
        freqs = 2*np.pi*np.ones(n_oscillators)
        rates = np.ones(n_oscillators)
        options = SalamanderControlOptions.walking()
        # Amplitudes
        amplitudes = np.zeros(n_oscillators)
        for i in range(n_body):
            amplitudes[[i, i+n_body]] = (
                options["body_stand_amplitude"]*np.sin(
                    2*np.pi*i/n_body
                    - options["body_stand_shift"]
                )
            )
        for leg_i in range(n_legs):
            for i in range(n_dof_legs):
                amplitudes[[
                    2*n_body + 2*leg_i*n_dof_legs + i,
                    2*n_body + 2*leg_i*n_dof_legs + i + n_dof_legs
                ]] = (
                    options["leg_{}_amplitude".format(i)]
                )
        return cls.from_parameters(freqs, rates, amplitudes)

    @classmethod
    def for_swimming(cls):
        """Parameters for swimming"""
        n_body = 11
        n_dof_legs = 3
        n_legs = 4
        n_joints = n_body + n_legs*n_dof_legs
        n_oscillators = 2*(n_joints)
        freqs = 2*np.pi*np.ones(n_oscillators)
        rates = np.ones(n_oscillators)
        amplitudes = np.zeros(n_oscillators)
        options = SalamanderControlOptions.swimming()
        body_amplitudes = np.linspace(
            options["body_amplitude_0"],
            options["body_amplitude_1"],
            n_body
        )
        for i in range(n_body):
            amplitudes[[i, i+n_body]] = body_amplitudes[i]
        for leg_i in range(n_legs):
            for i in range(n_dof_legs):
                amplitudes[[
                    2*n_body + 2*leg_i*n_dof_legs + i,
                    2*n_body + 2*leg_i*n_dof_legs + i + n_dof_legs
                ]] = (
                    options["leg_{}_amplitude".format(i)]
                )
        return cls.from_parameters(freqs, rates, amplitudes)

    @property
    def freqs(self):
        """Frequencies"""
        return self.array[0]

    @freqs.setter
    def freqs(self, value):
        """Frequencies"""
        self.array[0] = value

    @property
    def amplitudes_rates(self):
        """Amplitudes rates"""
        return self.array[1]

    @property
    def amplitudes_desired(self):
        """Amplitudes desired"""
        return self.array[2]


class ConnectivityArray(NetworkArray):
    """Connectivity array"""

    @classmethod
    def from_parameters(cls, connections, weights, desired_phases):
        """From each parameter"""
        return cls(np.stack([connections, weights, desired_phases], axis=1))

    @classmethod
    def for_walking(cls):
        """Parameters for walking"""
        n_body_joints = 11
        connectivity = []

        # Body
        for i in range(n_body_joints-1):
            # i - i
            connectivity.append([
                bodyjoint2index(joint_i=i, side=1),
                bodyjoint2index(joint_i=i, side=0),
                3e2, np.pi
            ])
            connectivity.append([
                bodyjoint2index(joint_i=i, side=0),
                bodyjoint2index(joint_i=i, side=1),
                3e2, np.pi
            ])
            # i - i+1
            connectivity.append([
                bodyjoint2index(joint_i=i+1, side=0),
                bodyjoint2index(joint_i=i, side=0),
                3e2, 0
            ])
            connectivity.append([
                bodyjoint2index(joint_i=i, side=0),
                bodyjoint2index(joint_i=i+1, side=0),
                3e2, 0
            ])
        # i+1 - i+1 (final)
        connectivity.append([
            bodyjoint2index(joint_i=n_body_joints-1, side=1),
            bodyjoint2index(joint_i=n_body_joints-1, side=0),
            3e2, np.pi
        ])
        connectivity.append([
            bodyjoint2index(joint_i=n_body_joints-1, side=0),
            bodyjoint2index(joint_i=n_body_joints-1, side=1),
            3e2, np.pi
        ])

        # Legs (internal)
        for leg_i in range(2):
            for side_i in range(2):
                # 0 - 0
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    3e2, np.pi
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    3e2, np.pi
                ])
                # 0 - 1
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    3e2, 0.5*np.pi
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    3e2, -0.5*np.pi
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    3e2, 0.5*np.pi
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    3e2, -0.5*np.pi
                ])
                # 1 - 1
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    3e2, np.pi
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    3e2, np.pi
                ])
                # 1 - 2
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    3e2, 0
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    3e2, 0
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    3e2, 0
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    3e2, 0
                ])
                # 2 - 2
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    3e2, np.pi
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    3e2, np.pi
                ])

        # Opposite leg interaction
        # TODO

        # Following leg interaction
        # TODO

        # Body-legs interaction
        for side_i in range(2):
            # Forelimbs
            connectivity.append([
                legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=0),
                bodyjoint2index(joint_i=1, side=side_i),
                3e2, 0
            ])
            connectivity.append([
                bodyjoint2index(joint_i=1, side=side_i),
                legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=0),
                3e2, 0
            ])
            connectivity.append([
                legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=1),
                bodyjoint2index(joint_i=1, side=side_i),
                3e2, np.pi
            ])
            connectivity.append([
                bodyjoint2index(joint_i=1, side=side_i),
                legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=1),
                3e2, -np.pi
            ])
            # Hind limbs
            connectivity.append([
                legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=0),
                bodyjoint2index(joint_i=4, side=side_i),
                3e2, np.pi
            ])
            connectivity.append([
                bodyjoint2index(joint_i=4, side=side_i),
                legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=0),
                3e2, -np.pi
            ])
            connectivity.append([
                legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=1),
                bodyjoint2index(joint_i=4, side=side_i),
                3e2, 0
            ])
            connectivity.append([
                bodyjoint2index(joint_i=4, side=side_i),
                legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=1),
                3e2, 0
            ])
        return cls(np.array(connectivity))

    @classmethod
    def for_swimming(cls):
        """Parameters for swimming"""
        n_body_joints = 11
        connectivity = []

        # Body
        for i in range(n_body_joints-1):
            # i - i
            connectivity.append([
                bodyjoint2index(joint_i=i, side=1),
                bodyjoint2index(joint_i=i, side=0),
                3e2, np.pi
            ])
            connectivity.append([
                bodyjoint2index(joint_i=i, side=0),
                bodyjoint2index(joint_i=i, side=1),
                3e2, np.pi
            ])
            # i - i+1
            connectivity.append([
                bodyjoint2index(joint_i=i+1, side=0),
                bodyjoint2index(joint_i=i, side=0),
                3e2, 2*np.pi/n_body_joints
            ])
            connectivity.append([
                bodyjoint2index(joint_i=i, side=0),
                bodyjoint2index(joint_i=i+1, side=0),
                3e2, -2*np.pi/n_body_joints
            ])
        # i+1 - i+1 (final)
        connectivity.append([
            bodyjoint2index(joint_i=n_body_joints-1, side=1),
            bodyjoint2index(joint_i=n_body_joints-1, side=0),
            3e2, np.pi
        ])
        connectivity.append([
            bodyjoint2index(joint_i=n_body_joints-1, side=0),
            bodyjoint2index(joint_i=n_body_joints-1, side=1),
            3e2, np.pi
        ])

        # Legs (internal)
        for leg_i in range(2):
            for side_i in range(2):
                # 0 - 0
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    3e2, np.pi
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    3e2, np.pi
                ])
                # 0 - 1
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    3e2, 0.5*np.pi
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    3e2, -0.5*np.pi
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    3e2, 0.5*np.pi
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    3e2, -0.5*np.pi
                ])
                # 1 - 1
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    3e2, np.pi
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    3e2, np.pi
                ])
                # 1 - 2
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    3e2, 0
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    3e2, 0
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    3e2, 0
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    3e2, 0
                ])
                # 2 - 2
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    3e2, 0
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    3e2, 0
                ])

        # Opposite leg interaction
        # TODO

        # Following leg interaction
        # TODO

        # Body-legs interaction
        for side_i in range(2):
            # Forelimbs
            connectivity.append([
                legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=0),
                bodyjoint2index(joint_i=1, side=side_i),
                3e2, 0
            ])
            connectivity.append([
                bodyjoint2index(joint_i=1, side=side_i),
                legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=0),
                0, 0
            ])
            connectivity.append([
                legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=1),
                bodyjoint2index(joint_i=1, side=side_i),
                3e2, np.pi
            ])
            connectivity.append([
                bodyjoint2index(joint_i=1, side=side_i),
                legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=1),
                0, -np.pi
            ])
            # Hind limbs
            connectivity.append([
                legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=0),
                bodyjoint2index(joint_i=4, side=side_i),
                3e2, np.pi
            ])
            connectivity.append([
                bodyjoint2index(joint_i=4, side=side_i),
                legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=0),
                0, -np.pi
            ])
            connectivity.append([
                legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=1),
                bodyjoint2index(joint_i=4, side=side_i),
                3e2, 0
            ])
            connectivity.append([
                bodyjoint2index(joint_i=4, side=side_i),
                legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=1),
                0, 0
            ])
        return cls(np.array(connectivity))

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


class JointsArray(NetworkArray):
    """Oscillator array"""

    @classmethod
    def from_parameters(cls, offsets):
        """From each parameter"""
        return cls(np.array([offsets]))

    @classmethod
    def for_walking(cls):
        """Parameters for walking"""
        n_body = 11
        n_dof_legs = 3
        n_legs = 4
        n_joints = n_body + n_legs*n_dof_legs
        options = SalamanderControlOptions.walking()
        offsets = np.zeros(n_joints)
        for leg_i in range(n_legs):
            for i in range(n_dof_legs):
                offsets[n_body + leg_i*n_dof_legs + i] = (
                    options["leg_{}_offset".format(i)]
                )
        return cls.from_parameters(offsets)

    @classmethod
    def for_swimming(cls):
        """Parameters for swimming"""
        n_body = 11
        n_dof_legs = 3
        n_legs = 4
        n_joints = n_body + n_legs*n_dof_legs
        options = SalamanderControlOptions.swimming()
        offsets = np.zeros(n_joints)
        for leg_i in range(n_legs):
            for i in range(n_dof_legs):
                offsets[n_body + leg_i*n_dof_legs + i] = (
                    options["leg_{}_offset".format(i)]
                )
        return cls.from_parameters(offsets)

    @property
    def offsets(self):
        """Joints anglers offsets"""
        return self.array[0]


class SalamanderNetworkODE(ODESolver):
    """Salamander network"""

    def __init__(self, state, parameters, timestep):
        super(SalamanderNetworkODE, self).__init__(
            ode=ODE(rk4, ode_oscillators_sparse),
            state=state.array,
            timestep=timestep,
            parameters=parameters.to_ode_parameters()
        )
        self.state = state
        self.parameters = parameters
        self._n_oscillators = state.n_oscillators
        n_body = 11
        n_legs_dofs = 3
        # n_legs = 4
        self.group0 = [
            bodyjoint2index(joint_i=i, side=0)
            for i in range(11)
        ] + [
            legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=joint_i, side=0)
            for leg_i in range(2)
            for side_i in range(2)
            for joint_i in range(n_legs_dofs)
        ]
        self.group1 = [
            bodyjoint2index(joint_i=i, side=1)
            for i in range(n_body)
        ] + [
            legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=joint_i, side=1)
            for leg_i in range(2)
            for side_i in range(2)
            for joint_i in range(n_legs_dofs)
        ]

    @classmethod
    def from_gait(cls, gait, n_iterations, timestep):
        """ Salamander network from gait"""
        return (
            cls.swimming(n_iterations, timestep)
            if gait == "swimming"
            else cls.walking(n_iterations, timestep)
        )

    @classmethod
    def walking(cls, n_iterations, timestep):
        """Salamander swimming network"""
        n_oscillators = 2*(11+4*3)
        state = OscillatorNetworkState.from_initial_state(
            initial_state=np.zeros(2*n_oscillators),
            n_iterations=n_iterations,
            n_oscillators=n_oscillators
        )
        parameters = SalamanderNetworkParameters.for_walking()
        return cls(state, parameters, timestep)

    @classmethod
    def swimming(cls, n_iterations, timestep):
        """Salamander swimming network"""
        n_oscillators = 2*(11+4*3)
        state = OscillatorNetworkState.from_initial_state(
            initial_state=np.zeros(2*n_oscillators),
            n_iterations=n_iterations,
            n_oscillators=n_oscillators
        )
        parameters = SalamanderNetworkParameters.for_swimming()
        return cls(state, parameters, timestep)

    def control_step(self, freqs):
        """Control step"""
        self.parameters.oscillators.freqs = freqs
        self.step()
        return self.current_state

    @property
    def phases(self):
        """Oscillators phases"""
        return self._state[:, 0, :self._n_oscillators]

    @property
    def dphases(self):
        """Oscillators phases velocity"""
        return self._state[:, 1, :self._n_oscillators]

    @property
    def amplitudes(self):
        """Amplitudes"""
        return self._state[:, 0, self._n_oscillators:]

    @property
    def damplitudes(self):
        """Amplitudes velocity"""
        return self._state[:, 1, self._n_oscillators:]

    def get_outputs(self):
        """Outputs"""
        return self.amplitudes[self.iteration]*(
            1 + np.cos(self.phases[self.iteration])
        )

    def get_outputs_all(self):
        """Outputs"""
        return self.amplitudes*(
            1 + np.cos(self.phases)
        )

    def get_doutputs(self):
        """Outputs velocity"""
        return self.damplitudes[self.iteration]*(
            1 + np.cos(self.phases[self.iteration])
        ) - (
            self.amplitudes[self.iteration]
            *np.sin(self.phases[self.iteration])
            *self.dphases[self.iteration]
        )

    def get_doutputs_all(self):
        """Outputs velocity"""
        return self.damplitudes*(
            1 + np.cos(self.phases)
        ) - self.amplitudes*np.sin(self.phases)*self.dphases

    def get_position_output(self):
        """Position output"""
        outputs = self.get_outputs()
        return (
            0.5*(outputs[self.group0] - outputs[self.group1])
            + self.parameters.joints.offsets
        )

    def get_position_output_all(self):
        """Position output"""
        outputs = self.get_outputs_all()
        return (
            0.5*(outputs[:, self.group0] - outputs[:, self.group1])
            + self.parameters.joints.offsets
        )

    def get_velocity_output(self):
        """Position output"""
        outputs = self.get_doutputs()
        return 0.5*(outputs[self.group0] - outputs[self.group1])

    def get_velocity_output_all(self):
        """Position output"""
        outputs = self.get_doutputs_all()
        return 0.5*(outputs[:, self.group0] - outputs[:, self.group1])
