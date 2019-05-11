"""Network"""

import numpy as np
from ..cy_controller import ode_oscillators_sparse, rk4, euler
from .convention import bodyjoint2index, legjoint2index
from .control_options import SalamanderControlOptions


class ODE(dict):
    """ODE"""

    def __init__(self, solver, function):
        super(ODE, self).__init__({"solver": solver, "function": function})

    @property
    def solver(self):
        """Solver"""
        return self["solver"]

    @property
    def function(self):
        """Function"""
        return self["function"]


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


class NetworkArray:
    """Network array"""

    def __init__(self, array):
        super(NetworkArray, self).__init__()
        self._array = array

    @property
    def array(self):
        """Array"""
        return self._array

    def shape(self):
        """Array shape"""
        return np.shape(self._array)


class OscillatorNetworkState(NetworkArray):
    """Network state"""

    def __init__(self, state, n_oscillators, iteration=0):
        self.n_oscillators = n_oscillators
        self._iteration = iteration
        super(OscillatorNetworkState, self).__init__(state)

    @staticmethod
    def default_initial_state():
        """Default state"""
        n_joints = 11+4*3
        return np.concatenate([
            np.linspace(1e-3, 0, 2*n_joints),
            np.zeros(2*n_joints),
            np.zeros(n_joints)
        ])

    @staticmethod
    def default_state(n_iterations):
        """Default state"""
        n_joints = 11+4*3
        n_oscillators = 2*n_joints
        return OscillatorNetworkState.from_initial_state(
            initial_state=OscillatorNetworkState.default_initial_state(),
            n_iterations=n_iterations,
            n_oscillators=n_oscillators
        )

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
            [NetworkArray(np.zeros([  # Runge-Kutta parameters
                7,
                2*oscillators.shape()[1] + 1*joints.shape()[1]
            ]))],
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

    def update_gait(self, gait):
        """Update from gait"""
        if gait == "walking":
            self.function[0:3] = [
                OscillatorArray.for_walking(),
                ConnectivityArray.for_walking(),
                JointsArray.for_walking()
            ]
        else:
            self.function[0:3] = [
                OscillatorArray.for_swimming(),
                ConnectivityArray.for_swimming(),
                JointsArray.for_swimming()
            ]

    @staticmethod
    def walking_parameters():
        """Walking parameters"""
        oscillators = OscillatorArray.for_walking()
        connectivity = ConnectivityArray.for_walking()
        joints = JointsArray.for_walking()
        return oscillators, connectivity, joints

    @staticmethod
    def swimming_parameters():
        """Swimming parameters"""
        oscillators = OscillatorArray.for_swimming()
        connectivity = ConnectivityArray.for_swimming()
        joints = JointsArray.for_swimming()
        return oscillators, connectivity, joints

    @classmethod
    def for_walking(cls):
        """Salamander swimming network"""
        oscillators, connectivity, joints = cls.walking_parameters()
        return cls(oscillators, connectivity, joints)

    @classmethod
    def for_swimming(cls):
        """Salamander swimming network"""
        oscillators, connectivity, joints = cls.swimming_parameters()
        return cls(oscillators, connectivity, joints)

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

    def to_ode_parameters(self):
        """Convert 2 arrays"""
        return ODE(
            [parameter.array for parameter in self.solver],
            [parameter.array for parameter in self.function]
            + [self.oscillators.shape()[1]]
            + [self.connectivity.shape()[0]]
            + [self.joints.shape()[1]]
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

    @staticmethod
    def walking_parameters():
        """Walking parameters"""
        n_body = 11
        n_dof_legs = 3
        n_legs = 4
        n_joints = n_body + n_legs*n_dof_legs
        n_oscillators = 2*(n_joints)
        freqs = 2*np.pi*np.ones(n_oscillators)
        rates = 10*np.ones(n_oscillators)
        options = SalamanderControlOptions.walking()
        # Amplitudes
        amplitudes = np.zeros(n_oscillators)
        for i in range(n_body):
            amplitudes[[i, i+n_body]] = options["body_stand_amplitude"]*np.sin(
                2*np.pi*i/n_body
                - options["body_stand_shift"]
            )
        for leg_i in range(n_legs):
            for i in range(n_dof_legs):
                amplitudes[[
                    2*n_body + 2*leg_i*n_dof_legs + i,
                    2*n_body + 2*leg_i*n_dof_legs + i + n_dof_legs
                ]] = options["leg_{}_amplitude".format(i)]
        # print("Amplitudes along body: abs({})".format(amplitudes[:11]))
        amplitudes = np.abs(amplitudes)
        return freqs, rates, amplitudes

    @staticmethod
    def swimming_parameters():
        """Swimming parameters"""
        n_body = 11
        n_dof_legs = 3
        n_legs = 4
        n_joints = n_body + n_legs*n_dof_legs
        n_oscillators = 2*(n_joints)
        freqs = 2*np.pi*np.ones(n_oscillators)
        rates = 10*np.ones(n_oscillators)
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
        return freqs, rates, amplitudes

    @classmethod
    def for_walking(cls):
        """Parameters for walking"""
        freqs, rates, amplitudes = cls.walking_parameters()
        return cls.from_parameters(freqs, rates, amplitudes)

    @classmethod
    def for_swimming(cls):
        """Parameters for swimming"""
        freqs, rates, amplitudes = cls.swimming_parameters()
        return cls.from_parameters(freqs, rates, amplitudes)

    @property
    def freqs(self):
        """Frequencies"""
        return 0.5*self.array[0]/np.pi

    @freqs.setter
    def freqs(self, value):
        """Frequencies"""
        self.array[0, :] = 2*np.pi*value

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

    def update_drives(self, drive_speed, drive_turn):
        """Set freqs"""
        self.freqs = 4*drive_speed*np.ones(self.shape()[1])
        self.amplitudes_desired = self._original_amplitudes_desired


class ConnectivityArray(NetworkArray):
    """Connectivity array"""

    @classmethod
    def from_parameters(cls, connections, weights, desired_phases):
        """From each parameter"""
        return cls(np.stack([connections, weights, desired_phases], axis=1))

    @staticmethod
    def walking_parameters():
        """Walking parameters"""
        n_body_joints = 11
        connectivity = []
        body_amplitude = 1e2
        legs_amplitude = 3e2
        legs2body_amplitude = 3e2

        # Amplitudes
        options = SalamanderControlOptions.walking()
        amplitudes = [
            options["body_stand_amplitude"]*np.sin(
                2*np.pi*i/n_body_joints
                - options["body_stand_shift"]
            )
            for i in range(n_body_joints)
        ]

        # Body
        for i in range(n_body_joints-1):
            # i - i
            connectivity.append([
                bodyjoint2index(joint_i=i, side=1),
                bodyjoint2index(joint_i=i, side=0),
                body_amplitude, np.pi
            ])
            connectivity.append([
                bodyjoint2index(joint_i=i, side=0),
                bodyjoint2index(joint_i=i, side=1),
                body_amplitude, np.pi
            ])
            # i - i+1
            phase_diff = (
                0
                if np.sign(amplitudes[i]) == np.sign(amplitudes[i+1])
                else np.pi
            )
            for side in range(2):
                connectivity.append([
                    bodyjoint2index(joint_i=i+1, side=side),
                    bodyjoint2index(joint_i=i, side=side),
                    body_amplitude, phase_diff
                ])
                connectivity.append([
                    bodyjoint2index(joint_i=i, side=side),
                    bodyjoint2index(joint_i=i+1, side=side),
                    body_amplitude, phase_diff
                ])
        # i+1 - i+1 (final)
        connectivity.append([
            bodyjoint2index(joint_i=n_body_joints-1, side=1),
            bodyjoint2index(joint_i=n_body_joints-1, side=0),
            body_amplitude, np.pi
        ])
        connectivity.append([
            bodyjoint2index(joint_i=n_body_joints-1, side=0),
            bodyjoint2index(joint_i=n_body_joints-1, side=1),
            body_amplitude, np.pi
        ])

        # Legs (internal)
        for leg_i in range(2):
            for side_i in range(2):
                # 0 - 0
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legs_amplitude, np.pi
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legs_amplitude, np.pi
                ])
                # 0 - 1
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legs_amplitude, 0.5*np.pi
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legs_amplitude, -0.5*np.pi
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legs_amplitude, 0.5*np.pi
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legs_amplitude, -0.5*np.pi
                ])
                # 1 - 1
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legs_amplitude, np.pi
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legs_amplitude, np.pi
                ])
                # 1 - 2
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legs_amplitude, 0
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    legs_amplitude, 0
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legs_amplitude, 0
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    legs_amplitude, 0
                ])
                # 2 - 2
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    legs_amplitude, np.pi
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    legs_amplitude, np.pi
                ])

        # Opposite leg interaction
        # TODO
        for leg_i in range(2):
            for side in range(2):
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=0, joint_i=0, side=side),
                    legjoint2index(leg_i=leg_i, side_i=1, joint_i=0, side=side),
                    legs_amplitude, np.pi
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=1, joint_i=0, side=side),
                    legjoint2index(leg_i=leg_i, side_i=0, joint_i=0, side=side),
                    legs_amplitude, np.pi
                ])

        # Following leg interaction
        # TODO
        for side_i in range(2):
            for side in range(2):
                connectivity.append([
                    legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=side),
                    legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=side),
                    legs_amplitude, np.pi
                ])
                connectivity.append([
                    legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=side),
                    legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=side),
                    legs_amplitude, np.pi
                ])

        # Body-legs interaction
        for side_i in range(2):
            for i in [0, 1, 7, 8, 9, 10]:
                # Forelimbs
                connectivity.append([
                    bodyjoint2index(joint_i=i, side=side_i),
                    legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=0),
                    legs2body_amplitude, np.pi
                ])
                # connectivity.append([
                #     bodyjoint2index(joint_i=1, side=side_i),
                #     legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=0),
                #     legs2body_amplitude, np.pi
                # ])
                connectivity.append([
                    bodyjoint2index(joint_i=i, side=side_i),
                    legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=1),
                    legs2body_amplitude, 0
                ])
                # connectivity.append([
                #     bodyjoint2index(joint_i=1, side=side_i),
                #     legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=1),
                #     legs2body_amplitude, 0
                # ])
            for i in [2, 3, 4, 5]:
                # Hind limbs
                connectivity.append([
                    bodyjoint2index(joint_i=i+4, side=side_i),
                    legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=0),
                    legs2body_amplitude, np.pi
                ])
                # connectivity.append([
                #     bodyjoint2index(joint_i=4, side=side_i),
                #     legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=0),
                #     legs2body_amplitude, np.pi
                # ])
                connectivity.append([
                    bodyjoint2index(joint_i=i+4, side=side_i),
                    legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=1),
                    legs2body_amplitude, 0
                ])
                # connectivity.append([
                #     bodyjoint2index(joint_i=4, side=side_i),
                #     legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=1),
                #     legs2body_amplitude, 0
                # ])
        return connectivity

    @staticmethod
    def swimming_parameters():
        """Swimming parameters"""
        n_body_joints = 11
        connectivity = []
        body_amplitude = 1e2
        legs_amplitude = 3e2
        legs2body_amplitude = 3e2

        # Body
        for i in range(n_body_joints-1):
            # i - i
            connectivity.append([
                bodyjoint2index(joint_i=i, side=1),
                bodyjoint2index(joint_i=i, side=0),
                body_amplitude, np.pi
            ])
            connectivity.append([
                bodyjoint2index(joint_i=i, side=0),
                bodyjoint2index(joint_i=i, side=1),
                body_amplitude, np.pi
            ])
            # i - i+1
            for side in range(2):
                connectivity.append([
                    bodyjoint2index(joint_i=i+1, side=side),
                    bodyjoint2index(joint_i=i, side=side),
                    body_amplitude, 2*np.pi/n_body_joints
                ])
                connectivity.append([
                    bodyjoint2index(joint_i=i, side=side),
                    bodyjoint2index(joint_i=i+1, side=side),
                    body_amplitude, -2*np.pi/n_body_joints
                ])
        # i+1 - i+1 (final)
        connectivity.append([
            bodyjoint2index(joint_i=n_body_joints-1, side=1),
            bodyjoint2index(joint_i=n_body_joints-1, side=0),
            body_amplitude, np.pi
        ])
        connectivity.append([
            bodyjoint2index(joint_i=n_body_joints-1, side=0),
            bodyjoint2index(joint_i=n_body_joints-1, side=1),
            body_amplitude, np.pi
        ])

        # Legs (internal)
        for leg_i in range(2):
            for side_i in range(2):
                # 0 - 0
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legs_amplitude, np.pi
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legs_amplitude, np.pi
                ])
                # 0 - 1
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legs_amplitude, 0.5*np.pi
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legs_amplitude, -0.5*np.pi
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legs_amplitude, 0.5*np.pi
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legs_amplitude, -0.5*np.pi
                ])
                # 1 - 1
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legs_amplitude, np.pi
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legs_amplitude, np.pi
                ])
                # 1 - 2
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legs_amplitude, 0
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    legs_amplitude, 0
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legs_amplitude, 0
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    legs_amplitude, 0
                ])
                # 2 - 2
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    legs_amplitude, 0
                ])
                connectivity.append([
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    legs_amplitude, 0
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
                legs2body_amplitude, np.pi
            ])
            connectivity.append([
                bodyjoint2index(joint_i=1, side=side_i),
                legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=0),
                legs2body_amplitude, np.pi
            ])
            connectivity.append([
                legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=1),
                bodyjoint2index(joint_i=1, side=side_i),
                legs2body_amplitude, 0
            ])
            connectivity.append([
                bodyjoint2index(joint_i=1, side=side_i),
                legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=1),
                legs2body_amplitude, 0
            ])
            # Hind limbs
            connectivity.append([
                legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=0),
                bodyjoint2index(joint_i=4, side=side_i),
                legs2body_amplitude, np.pi
            ])
            connectivity.append([
                bodyjoint2index(joint_i=4, side=side_i),
                legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=0),
                legs2body_amplitude, np.pi
            ])
            connectivity.append([
                legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=1),
                bodyjoint2index(joint_i=4, side=side_i),
                legs2body_amplitude, 0
            ])
            connectivity.append([
                bodyjoint2index(joint_i=4, side=side_i),
                legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=1),
                legs2body_amplitude, 0
            ])
        return connectivity

    @classmethod
    def for_walking(cls):
        """Parameters for walking"""
        connectivity = cls.walking_parameters()
        return cls(np.array(connectivity))

    @classmethod
    def for_swimming(cls):
        """Parameters for swimming"""
        connectivity = cls.swimming_parameters()
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
    def from_parameters(cls, offsets, rates):
        """From each parameter"""
        return cls(np.array([offsets, rates]))

    @staticmethod
    def walking_parameters():
        """Walking parameters"""
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
        rates = 5*np.ones(n_joints)
        return offsets, rates

    @staticmethod
    def swimming_parameters():
        """Swimming parameters"""
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
        rates = 5*np.ones(n_joints)
        return offsets, rates

    @classmethod
    def for_walking(cls):
        """Parameters for walking"""
        offsets, rates = cls.walking_parameters()
        return cls.from_parameters(offsets, rates)

    @classmethod
    def for_swimming(cls):
        """Parameters for swimming"""
        offsets, rates = cls.swimming_parameters()
        return cls.from_parameters(offsets, rates)

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
        self._n_joints = parameters.joints.shape()[1]
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

    def update_gait(self, gait):
        """Update from gait"""
        self.parameters.update_gait(gait)
        self._parameters = self.parameters.to_ode_parameters()

    @classmethod
    def walking(cls, n_iterations, timestep):
        """Salamander swimming network"""
        state = OscillatorNetworkState.default_state(n_iterations)
        parameters = SalamanderNetworkParameters.for_walking()
        return cls(state, parameters, timestep)

    @classmethod
    def swimming(cls, n_iterations, timestep):
        """Salamander swimming network"""
        state = OscillatorNetworkState.default_state(n_iterations)
        parameters = SalamanderNetworkParameters.for_swimming()
        return cls(state, parameters, timestep)

    def control_step(self):
        """Control step"""
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
        return self._state[:, 0, self._n_oscillators:2*self._n_oscillators]

    @property
    def damplitudes(self):
        """Amplitudes velocity"""
        return self._state[:, 1, self._n_oscillators:2*self._n_oscillators]

    @property
    def offsets(self):
        """Offset"""
        return self._state[:, 0, 2*self._n_oscillators:]

    @property
    def doffsets(self):
        """Offset velocity"""
        return self._state[:, 1, 2*self._n_oscillators:]

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
            + self.offsets[self.iteration]
        )

    def get_position_output_all(self):
        """Position output"""
        outputs = self.get_outputs_all()
        return (
            0.5*(outputs[:, self.group0] - outputs[:, self.group1])
            + self.offsets
        )

    def get_velocity_output(self):
        """Position output"""
        outputs = self.get_doutputs()
        return 0.5*(outputs[self.group0] - outputs[self.group1])

    def get_velocity_output_all(self):
        """Position output"""
        outputs = self.get_doutputs_all()
        return 0.5*(outputs[:, self.group0] - outputs[:, self.group1])

    def update_drive(self, drive_speed, drive_turn):
        """Update drives"""
        print("TODO: Update drives to speed={} and turn={}".format(
            drive_speed,
            drive_turn
        ))
        self.parameters.oscillators.update_drives(drive_speed, drive_turn)
