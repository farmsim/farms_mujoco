"""Network"""

import numpy as np
from ..cy_controller import odefun_sparse, rk4_ode_sparse
from .convention import bodyjoint2index, legjoint2index


class Network:
    """Controller network"""

    def __init__(self, ode, state, ode_solver, timestep):
        super(Network, self).__init__()
        self._ode = ode
        self._state = state
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

    def integrate(self, *parameters):
        """Control step"""
        self._ode_solver(
            self._ode,
            self._timestep,
            self._state,
            *parameters
        )
        self._time += self._timestep


class SalamanderNetwork(Network):
    """Salamander network"""

    def __init__(self, phases, freqs, connectivity, timestep):
        self._freqs = freqs
        self._connectivity = np.array(connectivity[:, :2], dtype=np.uintc)
        self._connections = np.array(connectivity[:, 2:], dtype=np.float64)
        self._n_dim = np.shape(self._freqs)[0]
        self._c_dim = np.shape(self._connectivity)[0]
        super(SalamanderNetwork, self).__init__(
            ode=odefun_sparse,
            state=phases,
            ode_solver=rk4_ode_sparse,
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

    # @classmethod
    # def swimming(cls, timestep, phases=None):
    #     """Default salamander network"""
    #     n_dim_body = 11
    #     n_dim_legs = 2*2*3
    #     n_dim = n_dim_body + n_dim_legs
    #     weights = np.zeros([n_dim, n_dim])
    #     phases_desired = np.zeros([n_dim, n_dim])
    #     # Body
    #     for i in range(10):
    #         weights[i, i+1] = 3e2
    #         weights[i+1, i] = 3e2
    #         phases_desired[i, i+1] = 2*np.pi/n_dim_body
    #         phases_desired[i+1, i] = -2*np.pi/n_dim_body
    #     # Legs
    #     for leg_i in range(2):
    #         for side_i in range(2):
    #             # 0 - 1
    #             weights[
    #                 legjoint2index(leg_i, side_i, 0),
    #                 legjoint2index(leg_i, side_i, 1)
    #             ] = 3e2
    #             weights[
    #                 legjoint2index(leg_i, side_i, 1),
    #                 legjoint2index(leg_i, side_i, 0)
    #             ] = 3e2
    #             phases_desired[
    #                 legjoint2index(leg_i, side_i, 0),
    #                 legjoint2index(leg_i, side_i, 1)
    #             ] = 0
    #             phases_desired[
    #                 legjoint2index(leg_i, side_i, 1),
    #                 legjoint2index(leg_i, side_i, 0)
    #             ] = 0
    #             # 1 - 2
    #             weights[
    #                 legjoint2index(leg_i, side_i, 1),
    #                 legjoint2index(leg_i, side_i, 2)
    #             ] = 3e2
    #             weights[
    #                 legjoint2index(leg_i, side_i, 2),
    #                 legjoint2index(leg_i, side_i, 1)
    #             ] = 3e2
    #             phases_desired[
    #                 legjoint2index(leg_i, side_i, 1),
    #                 legjoint2index(leg_i, side_i, 2)
    #             ] = 0
    #             phases_desired[
    #                 legjoint2index(leg_i, side_i, 2),
    #                 legjoint2index(leg_i, side_i, 1)
    #             ] = 0
    #     # # Opposite leg interaction
    #     # for leg_i in range(2):
    #     #     # 0 - 1
    #     #     weights[
    #     #         legjoint2index(leg_i, 0, 0),
    #     #         legjoint2index(leg_i, 1, 0)
    #     #     ] = 3e2
    #     #     weights[
    #     #         legjoint2index(leg_i, 1, 0),
    #     #         legjoint2index(leg_i, 0, 0)
    #     #     ] = 3e2
    #     #     phases_desired[
    #     #         legjoint2index(leg_i, 0, 0),
    #     #         legjoint2index(leg_i, 1, 0)
    #     #     ] = np.pi
    #     #     phases_desired[
    #     #         legjoint2index(leg_i, 1, 0),
    #     #         legjoint2index(leg_i, 0, 0)
    #     #     ] = -np.pi
    #     # # Following leg interaction
    #     # for side_i in range(2):
    #     #     # 0 - 1
    #     #     weights[
    #     #         legjoint2index(0, side_i, 0),
    #     #         legjoint2index(1, side_i, 0)
    #     #     ] = 3e2
    #     #     weights[
    #     #         legjoint2index(1, side_i, 0),
    #     #         legjoint2index(0, side_i, 0)
    #     #     ] = 3e2
    #     #     phases_desired[
    #     #         legjoint2index(0, side_i, 0),
    #     #         legjoint2index(1, side_i, 0)
    #     #     ] = np.pi
    #     #     phases_desired[
    #     #         legjoint2index(1, side_i, 0),
    #     #         legjoint2index(0, side_i, 0)
    #     #     ] = -np.pi
    #     # Body-legs interaction
    #     for side_i in range(2):
    #         # Forelimbs
    #         weights[
    #             bodyjoint2index(1),
    #             legjoint2index(0, side_i, 0)
    #         ] = 3e2
    #         weights[
    #             legjoint2index(0, side_i, 0),
    #             bodyjoint2index(1)
    #         ] = 3e2
    #         phases_desired[
    #             bodyjoint2index(1),
    #             legjoint2index(0, side_i, 0)
    #         ] = 0  # 0.5*np.pi
    #         phases_desired[
    #             legjoint2index(0, side_i, 0),
    #             bodyjoint2index(1)
    #         ] = 0  # -0.5*np.pi
    #         # Hind limbs
    #         weights[
    #             bodyjoint2index(4),
    #             legjoint2index(1, side_i, 0)
    #         ] = 3e2
    #         weights[
    #             legjoint2index(1, side_i, 0),
    #             bodyjoint2index(4)
    #         ] = 3e2
    #         phases_desired[
    #             bodyjoint2index(4),
    #             legjoint2index(1, side_i, 0)
    #         ] = 0  # -0.5*np.pi
    #         phases_desired[
    #             legjoint2index(1, side_i, 0),
    #             bodyjoint2index(4)
    #         ] = 0  # 0.5*np.pi
    #     freqs = 2*np.pi*np.ones(n_dim_body)
    #     if phases is None:
    #         phases = 1e-3*3e-1*np.pi*(2*np.pi*np.random.ranf(n_dim)-1)
    #     weights, phase_desired, integrator = cls.gen_cas_integrator(
    #         timestep,
    #         n_dim,
    #         weights,
    #         phases_desired
    #     )
    #     return cls(phases, freqs, weights, phase_desired, integrator)

    @property
    def phases(self):
        """Oscillator phases"""
        return self._state

    def control_step(self, freqs):
        """Control step"""
        self._freqs = np.array(freqs, dtype=np.float64)
        self.integrate(
            self._freqs,
            self._connectivity,
            self._connections,
            self._n_dim,
            self._c_dim
        )
        return self._state
