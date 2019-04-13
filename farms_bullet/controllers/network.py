"""Network"""

import numpy as np
from ..cy_controller import ode_oscillators_sparse, ode_radius, rk4
from .convention import bodyjoint2index, legjoint2index


class ODESolver:
    """Controller network"""

    def __init__(self, ode, state, ode_solver, timestep):
        super(ODESolver, self).__init__()
        self._ode = ode
        self._state = state
        self._n_dim = len(state)
        self._ode_solver = ode_solver
        self._time = 0
        self._timestep = timestep

    @property
    def state(self):
        """State"""
        return self._state

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


class SalamanderODERadius(ODESolver):
    """Salamander network"""

    def __init__(self, radius, rate, radius_desired, timestep):
        self._rate = np.array(rate, dtype=np.float64)
        self._radius_desired = np.array(radius_desired, dtype=np.float64)
        super(SalamanderODERadius, self).__init__(
            ode=ode_radius,
            state=radius,
            ode_solver=rk4,
            timestep=timestep
        )

    @classmethod
    def from_gait(cls, gait, timestep, radius=None):
        """ Salamander network from gait"""
        return (
            cls.swimming(timestep, radius)
            if gait == "swimming"
            else cls.walking(timestep, radius)
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
        radius = np.zeros(n_dim)
        rate = np.ones(n_dim)
        radius_desired = np.ones(n_dim)
        return radius, rate, radius_desired

    @classmethod
    def walking(cls, timestep, radius=None):
        """Default salamander network"""
        _radius, freqs, connectivity = (
            cls.walking_parameters()
        )
        if radius is None:
            radius = _radius
        return cls(radius, freqs, connectivity, timestep)

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
        radius = np.zeros(n_dim)
        rate = np.ones(n_dim)
        radius_desired = np.ones(n_dim)
        return radius, rate, radius_desired

    @classmethod
    def swimming(cls, timestep, radius=None):
        """Default salamander network"""
        radius, rate, radius_desired = cls.swimming_parameters()
        return cls(radius, rate, radius_desired, timestep)

    @property
    def radius(self):
        """Oscillator radius"""
        return self._state

    def control_step(self):
        """Control step"""
        self.integrate(
            [
                self._rate,
                self._radius_desired,
            ]
        )
        return self._state


class SalamanderNetwork:
    """Salamander network"""

    def __init__(self, phases_ode, radius_ode, offsets):
        super(SalamanderNetwork, self).__init__()
        self._ode_phases = phases_ode
        self._ode_radius = radius_ode
        self._offsets = offsets

    @classmethod
    def from_gait(cls, gait, timestep, phases=None, radius=None):
        """ Salamander network from gait"""
        return (
            cls.swimming(timestep, phases, radius)
            if gait == "swimming"
            else cls.walking(timestep, phases, radius)
        )

    @classmethod
    def walking(cls, timestep, phases=None, radius=None):
        """Network for walking"""
        phases_ode = SalamanderODEPhase.walking(timestep, phases)
        radius_ode = SalamanderODERadius.walking(timestep, radius)
        offsets = np.zeros(11+4*3)
        return cls(phases_ode, radius_ode, offsets)

    @classmethod
    def swimming(cls, timestep, phases=None, radius=None):
        """Network for """
        phases_ode = SalamanderODEPhase.swimming(timestep, phases)
        radius_ode = SalamanderODERadius.swimming(timestep, radius)
        offsets = np.zeros(11+4*3)
        return cls(phases_ode, radius_ode, offsets)

    def control_step(self, freqs):
        """Control step"""
        # return (
        #     self._ode_phases.control_step(freqs),
        #     self._ode_radius.control_step()
        # )
        self._ode_radius.control_step()
        return self._ode_phases.control_step(freqs)

    @property
    def phases(self):
        """Oscillators phases"""
        return self._ode_phases.state

    @property
    def radius(self):
        """Radius"""
        return self._ode_radius.state


class SalamanderNetworkPosition(SalamanderNetwork):
    """Salamander network for position control"""

    def get_position_output(self):
        """Position output"""
        phases, radius = self.phases, self.radius
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
        return np.concatenante([
            0.5*(
                radius[group0]*(1 + np.cos(phases[group0]))
                + radius[group1]*(1 + np.cos(phases[group1]))
            )
            
        ]) + self._offsets


# class SalamanderNetworkTorque(SalamanderNetwork):
#     """Salamander network for torque control"""

#     def get_torque_output(self):
#         """Position output"""
#         phases, radius = self.phases, self.radius
#         n_body = 11
#         return 0.5*(
#             radius[:n_body]*np.cos(1 + phases[:n_body])
#             + radius[:n_body]*np.cos(1 + phases[:n_body])
#         )
