"""Network"""

import numpy as np
from ..cy_controller import ode_oscillators_sparse, ode_amplitude, rk4
from .convention import bodyjoint2index, legjoint2index


class ODESolver:
    """Controller network"""

    def __init__(self, ode, state, ode_solver, timestep):
        super(ODESolver, self).__init__()
        self._ode = ode
        self._state = state
        self._dstate = np.copy(state)
        self._n_dim = len(state)
        self._ode_solver = ode_solver
        self._time = 0
        self._timestep = timestep

    @property
    def state(self):
        """State"""
        return self._state

    @property
    def dstate(self):
        """State"""
        return self._dstate

    @property
    def time(self):
        """Time"""
        return self._time

    def integrate(self, parameters):
        """Control step"""
        self._ode_solver(
            self._ode,
            self._timestep,
            self._state,
            self._dstate,
            self._n_dim,
            parameters
        )
        self._time += self._timestep


class SalamanderODEPhase(ODESolver):
    """Salamander network"""

    def __init__(self, phases, freqs, connectivity, timestep):
        self._freqs = freqs
        self._connectivity = np.array(connectivity[:, :2], dtype=np.uintc)
        self._connections = np.array(connectivity[:, 2:], dtype=np.float64)
        self._c_dim = np.shape(self._connectivity)[0]
        super(SalamanderODEPhase, self).__init__(
            ode=ode_oscillators_sparse,
            state=phases,
            ode_solver=rk4,
            timestep=timestep
        )

    @classmethod
    def from_gait(cls, gait, timestep, phases=None):
        """ Salamander network from gait"""
        return (
            cls.swimming(timestep, phases)
            if gait == "swimming"
            else cls.walking(timestep, phases)
        )

    @staticmethod
    def walking_parameters():
        """Walking parameters"""
        n_body_joints = 11
        n_sides = 2
        n_leg_pairs = 2
        n_leg_dof = 3
        n_dim_body = 2*n_body_joints
        n_dim_legs = 2*n_leg_pairs*n_sides*n_leg_dof
        n_dim = n_dim_body + n_dim_legs
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

        freqs = 2*np.pi*np.ones(n_dim)
        phases = 1e-3*3e-1*np.pi*(2*np.pi*np.random.ranf(n_dim)-1)
        return phases, freqs, np.array(connectivity)

    @classmethod
    def walking(cls, timestep, phases=None):
        """Default salamander network"""
        _phases, freqs, connectivity = (
            cls.walking_parameters()
        )
        if phases is None:
            phases = _phases
        return cls(phases, freqs, connectivity, timestep)

    @staticmethod
    def swimming_parameters():
        """Swimming parameters"""
        n_body_joints = 11
        n_sides = 2
        n_leg_pairs = 2
        n_leg_dof = 3
        n_dim_body = 2*n_body_joints
        n_dim_legs = 2*n_leg_pairs*n_sides*n_leg_dof
        n_dim = n_dim_body + n_dim_legs
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

        freqs = 2*np.pi*np.ones(n_dim)
        phases = 1e-3*3e-1*np.pi*(2*np.pi*np.random.ranf(n_dim)-1)
        return phases, freqs, np.array(connectivity)

    @classmethod
    def swimming(cls, timestep, phases=None):
        """Default salamander network"""
        _phases, freqs, connectivity = (
            cls.swimming_parameters()
        )
        if phases is None:
            phases = _phases
        return cls(phases, freqs, connectivity, timestep)

    @property
    def phases(self):
        """Oscillator phases"""
        return self._state

    def control_step(self, freqs):
        """Control step"""
        self._freqs = np.array(freqs, dtype=np.float64)
        self.integrate(
            [
                self._freqs,
                self._connectivity,
                self._connections,
                self._c_dim
            ]
        )
        return self._state


class SalamanderODEAmplitude(ODESolver):
    """Salamander network"""

    def __init__(self, amplitude, rate, amplitude_desired, timestep):
        self._rate = np.array(rate, dtype=np.float64)
        self._amplitude_desired = np.array(amplitude_desired, dtype=np.float64)
        super(SalamanderODEAmplitude, self).__init__(
            ode=ode_amplitude,
            state=amplitude,
            ode_solver=rk4,
            timestep=timestep
        )

    @classmethod
    def from_gait(cls, gait, timestep, amplitude=None):
        """ Salamander network from gait"""
        return (
            cls.swimming(timestep, amplitude)
            if gait == "swimming"
            else cls.walking(timestep, amplitude)
        )

    @staticmethod
    def walking_parameters():
        """Walking parameters"""
        n_body_joints = 11
        n_sides = 2
        n_leg_pairs = 2
        n_leg_dof = 3
        n_dim_body = 2*n_body_joints
        n_dim_legs = 2*n_leg_pairs*n_sides*n_leg_dof
        n_dim = n_dim_body + n_dim_legs
        amplitude = np.zeros(n_dim)
        rate = np.ones(n_dim)
        amplitude_desired = np.ones(n_dim)
        return amplitude, rate, amplitude_desired

    @classmethod
    def walking(cls, timestep, amplitude=None):
        """Default salamander network"""
        _amplitude, freqs, connectivity = (
            cls.walking_parameters()
        )
        if amplitude is None:
            amplitude = _amplitude
        return cls(amplitude, freqs, connectivity, timestep)

    @staticmethod
    def swimming_parameters():
        """Swimming parameters"""
        n_body_joints = 11
        n_sides = 2
        n_leg_pairs = 2
        n_leg_dof = 3
        n_dim_body = 2*n_body_joints
        n_dim_legs = 2*n_leg_pairs*n_sides*n_leg_dof
        n_dim = n_dim_body + n_dim_legs
        amplitude = np.zeros(n_dim)
        rate = np.ones(n_dim)
        amplitude_desired = np.ones(n_dim)
        return amplitude, rate, amplitude_desired

    @classmethod
    def swimming(cls, timestep, amplitude=None):
        """Default salamander network"""
        amplitude, rate, amplitude_desired = cls.swimming_parameters()
        return cls(amplitude, rate, amplitude_desired, timestep)

    @property
    def amplitude(self):
        """Oscillator amplitude"""
        return self._state

    def control_step(self):
        """Control step"""
        self.integrate(
            [
                self._rate,
                self._amplitude_desired,
            ]
        )
        return self._state


class SalamanderNetwork:
    """Salamander network"""

    def __init__(self, phases_ode, amplitude_ode):
        super(SalamanderNetwork, self).__init__()
        self._ode_phases = phases_ode
        self._ode_amplitude = amplitude_ode

    @classmethod
    def from_gait(cls, gait, timestep, phases=None, amplitude=None):
        """ Salamander network from gait"""
        return (
            cls.swimming(timestep, phases, amplitude)
            if gait == "swimming"
            else cls.walking(timestep, phases, amplitude)
        )

    @classmethod
    def walking(cls, timestep, phases=None, amplitude=None):
        """Network for walking"""
        phases_ode = SalamanderODEPhase.walking(timestep, phases)
        amplitude_ode = SalamanderODEAmplitude.walking(timestep, amplitude)
        return cls(phases_ode, amplitude_ode)

    @classmethod
    def swimming(cls, timestep, phases=None, amplitude=None):
        """Network for """
        phases_ode = SalamanderODEPhase.swimming(timestep, phases)
        amplitude_ode = SalamanderODEAmplitude.swimming(timestep, amplitude)
        return cls(phases_ode, amplitude_ode)

    def control_step(self, freqs):
        """Control step"""
        # return (
        #     self._ode_phases.control_step(freqs),
        #     self._ode_amplitude.control_step()
        # )
        self._ode_amplitude.control_step()
        return self._ode_phases.control_step(freqs)

    @property
    def phases(self):
        """Oscillators phases"""
        return self._ode_phases.state

    @property
    def dphases(self):
        """Oscillators phases velocity"""
        return self._ode_phases.dstate

    @property
    def amplitudes(self):
        """Amplitudes"""
        return self._ode_amplitude.state

    @property
    def damplitudes(self):
        """Amplitudes velocity"""
        return self._ode_amplitude.dstate

    def get_outputs(self):
        """Outputs"""
        return self.amplitudes*(
            1 + np.cos(2*np.pi*self.phases)
        )

    def get_velocity_outputs(self):
        """Outputs velocity"""
        return self.damplitudes*(
            1 + np.cos(2*np.pi*self.phases)
        ) - self.amplitudes*np.sin(2*np.pi*self.phases)*2*np.pi*self.dphases


class SalamanderNetworkPosition(SalamanderNetwork):
    """Salamander network for position control"""

    def __init__(self, phases_ode, amplitude_ode, offsets):
        super(SalamanderNetworkPosition, self).__init__(
            phases_ode,
            amplitude_ode
        )
        self._offsets = offsets

    @classmethod
    def pos_walking(cls, timestep, phases=None, amplitude=None, **kwargs):
        """Network for walking"""
        phases_ode = SalamanderODEPhase.walking(timestep, phases)
        amplitude_ode = SalamanderODEAmplitude.walking(timestep, amplitude)
        offsets = kwargs.pop("offsets", np.zeros(11+4*3))
        return cls(phases_ode, amplitude_ode, offsets)

    @classmethod
    def pos_swimming(cls, timestep, phases=None, amplitude=None, **kwargs):
        """Network for """
        phases_ode = SalamanderODEPhase.swimming(timestep, phases)
        amplitude_ode = SalamanderODEAmplitude.swimming(timestep, amplitude)
        offsets = kwargs.pop("offsets", np.zeros(11+4*3))
        return cls(phases_ode, amplitude_ode, offsets)

    def get_position_output(self):
        """Position output"""
        n_body = 11
        n_legs_dofs = 3
        n_legs = 4
        group0 = [i for i in range(11)] + [
            2*n_body + i+2*n_legs_dofs*j
            for i in range(n_legs_dofs)
            for j in range(n_legs)
        ]
        group1 = [n_body + i for i in range(11)] + [
            2*n_body + n_legs_dofs + i+2*n_legs_dofs*j
            for i in range(n_legs_dofs)
            for j in range(n_legs)
        ]
        outputs = self.get_outputs()
        return 0.5*(outputs[group0] - outputs[group1]) + self._offsets

    def get_velocity_output(self):
        """Position output"""
        n_body = 11
        n_legs_dofs = 3
        n_legs = 4
        group0 = [i for i in range(11)] + [
            2*n_body + i+2*n_legs_dofs*j
            for i in range(n_legs_dofs)
            for j in range(n_legs)
        ]
        group1 = [n_body + i for i in range(11)] + [
            2*n_body + n_legs_dofs + i+2*n_legs_dofs*j
            for i in range(n_legs_dofs)
            for j in range(n_legs)
        ]
        outputs = self.get_velocity_outputs()
        return 0.5*(outputs[group0] - outputs[group1])
