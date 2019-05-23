"""Network"""

import numpy as np
from scipy import integrate
from .convention import bodyosc2index, legosc2index  # legjoint2index
from ...controllers.network import (
    ODE,
    ODESolver,
    OscillatorNetworkState,
    NetworkParameters,
    OscillatorArray,
    ConnectivityArray,
    JointsArray
)
from ...cy_controller import (
    rk4,
    # euler,
    ode_oscillators_sparse,
    ode_oscillators_sparse_gradient
)
from .animat_options import SalamanderOptions, SalamanderControlOptions


class SalamanderOscillatorNetworkState(OscillatorNetworkState):
    """Network state"""

    @staticmethod
    def default_initial_state():
        """Default state"""
        n_dof_legs = 4
        n_joints = 11+4*n_dof_legs
        return np.concatenate([
            np.linspace(1e-3, 0, 2*n_joints),
            np.zeros(2*n_joints),
            np.zeros(n_joints)
        ])

    @staticmethod
    def default_state(n_iterations):
        """Default state"""
        n_dof_legs = 4
        n_joints = 11+4*n_dof_legs
        n_oscillators = 2*n_joints
        return SalamanderOscillatorNetworkState.from_initial_state(
            initial_state=(
                SalamanderOscillatorNetworkState.default_initial_state()
            ),
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


class SalamanderNetworkParameters(NetworkParameters):
    """Salamander network parameter"""

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
                SalamanderOscillatorArray.for_walking(),
                SalamanderConnectivityArray.for_walking(),
                SalamanderJointsArray.for_walking()
            ]
        else:
            self.function[0:3] = [
                SalamanderOscillatorArray.for_swimming(),
                SalamanderConnectivityArray.for_swimming(),
                SalamanderJointsArray.for_swimming()
            ]

    @staticmethod
    def walking_parameters():
        """Walking parameters"""
        oscillators = SalamanderOscillatorArray.for_walking()
        connectivity = SalamanderConnectivityArray.for_walking()
        joints = SalamanderJointsArray.for_walking()
        return oscillators, connectivity, joints

    @staticmethod
    def swimming_parameters():
        """Swimming parameters"""
        oscillators = SalamanderOscillatorArray.for_swimming()
        connectivity = SalamanderConnectivityArray.for_swimming()
        joints = SalamanderJointsArray.for_swimming()
        return oscillators, connectivity, joints

    @classmethod
    def default(cls):
        """Default salamander newtwork parameters"""
        oscillators = SalamanderOscillatorArray.default()
        connectivity = SalamanderConnectivityArray.default()
        joints = SalamanderJointsArray.default()
        return cls(oscillators, connectivity, joints)

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

    def update(self, parameters):
        """Update from gait"""
        self.function[0:3] = [
            SalamanderOscillatorArray.update(parameters),
            SalamanderConnectivityArray.update(parameters),
            SalamanderJointsArray.update(parameters)
        ]


class SalamanderOscillatorArray(OscillatorArray):
    """Oscillator array"""

    @staticmethod
    def default_parameters():
        """Walking parameters"""
        n_body = 11
        n_dof_legs = 4
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
    def walking_parameters():
        """Walking parameters"""
        n_body = 11
        n_dof_legs = 4
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
        n_dof_legs = 4
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
    def default(cls):
        """Default"""
        freqs, rates, amplitudes = cls.default_parameters()
        return cls.from_parameters(freqs, rates, amplitudes)

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


    def update_drives(self, drive_speed, drive_turn):
        """
        :param drive_speed: drive that change the frequency
        :param drive_turn: drive that change the offset
        :return: send to the simulation the drive
        """
        self.freq_sat_limb(drive_speed, drive_turn)
        self.freq_sat_body(drive_speed, drive_turn)
        self.amp_sat_body(drive_speed, drive_turn)
        self.amp_sat_limb(drive_speed, drive_turn)


class SalamanderConnectivityArray(ConnectivityArray):
    """Connectivity array"""

    @staticmethod
    def default_parameters():
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
                bodyosc2index(joint_i=i, side=1),
                bodyosc2index(joint_i=i, side=0),
                body_amplitude, np.pi
            ])
            connectivity.append([
                bodyosc2index(joint_i=i, side=0),
                bodyosc2index(joint_i=i, side=1),
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
                    bodyosc2index(joint_i=i+1, side=side),
                    bodyosc2index(joint_i=i, side=side),
                    body_amplitude, phase_diff
                ])
                connectivity.append([
                    bodyosc2index(joint_i=i, side=side),
                    bodyosc2index(joint_i=i+1, side=side),
                    body_amplitude, phase_diff
                ])
        # i+1 - i+1 (final)
        connectivity.append([
            bodyosc2index(joint_i=n_body_joints-1, side=1),
            bodyosc2index(joint_i=n_body_joints-1, side=0),
            body_amplitude, np.pi
        ])
        connectivity.append([
            bodyosc2index(joint_i=n_body_joints-1, side=0),
            bodyosc2index(joint_i=n_body_joints-1, side=1),
            body_amplitude, np.pi
        ])

        # Legs (internal)
        for leg_i in range(2):
            for side_i in range(2):
                # 0 - 0
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legs_amplitude, np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legs_amplitude, np.pi
                ])
                # 0 - 1
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legs_amplitude, 0.5*np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legs_amplitude, -0.5*np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legs_amplitude, 0.5*np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legs_amplitude, -0.5*np.pi
                ])
                # 1 - 1
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legs_amplitude, np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legs_amplitude, np.pi
                ])
                # 0 - 2
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legs_amplitude, 0
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    legs_amplitude, 0
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legs_amplitude, 0
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    legs_amplitude, 0
                ])
                # 2 - 2
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    legs_amplitude, np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    legs_amplitude, np.pi
                ])
                # 1 - 3
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legs_amplitude, 0
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=0),
                    legs_amplitude, 0
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legs_amplitude, 0
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=1),
                    legs_amplitude, 0
                ])
                # 3 - 3
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=0),
                    legs_amplitude, np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=1),
                    legs_amplitude, np.pi
                ])

        # Opposite leg interaction
        # TODO
        for leg_i in range(2):
            for side in range(2):
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=0, joint_i=0, side=side),
                    legosc2index(leg_i=leg_i, side_i=1, joint_i=0, side=side),
                    legs_amplitude, np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=1, joint_i=0, side=side),
                    legosc2index(leg_i=leg_i, side_i=0, joint_i=0, side=side),
                    legs_amplitude, np.pi
                ])

        # Following leg interaction
        # TODO
        for side_i in range(2):
            for side in range(2):
                connectivity.append([
                    legosc2index(leg_i=0, side_i=side_i, joint_i=0, side=side),
                    legosc2index(leg_i=1, side_i=side_i, joint_i=0, side=side),
                    legs_amplitude, np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=1, side_i=side_i, joint_i=0, side=side),
                    legosc2index(leg_i=0, side_i=side_i, joint_i=0, side=side),
                    legs_amplitude, np.pi
                ])

        # Body-legs interaction
        for side_i in range(2):
            for i in [0, 1, 7, 8, 9, 10]:
                # Forelimbs
                connectivity.append([
                    bodyosc2index(joint_i=i, side=side_i),
                    legosc2index(leg_i=0, side_i=side_i, joint_i=0, side=0),
                    legs2body_amplitude, 0
                ])
                # connectivity.append([
                #     bodyosc2index(joint_i=1, side=side_i),
                #     legosc2index(leg_i=0, side_i=side_i, joint_i=0, side=0),
                #     legs2body_amplitude, np.pi
                # ])
                connectivity.append([
                    bodyosc2index(joint_i=i, side=side_i),
                    legosc2index(leg_i=0, side_i=side_i, joint_i=0, side=1),
                    legs2body_amplitude, np.pi
                ])
                # connectivity.append([
                #     bodyosc2index(joint_i=1, side=side_i),
                #     legosc2index(leg_i=0, side_i=side_i, joint_i=0, side=1),
                #     legs2body_amplitude, 0
                # ])
            for i in [2, 3, 4, 5]:
                # Hind limbs
                connectivity.append([
                    bodyosc2index(joint_i=i+4, side=side_i),
                    legosc2index(leg_i=1, side_i=side_i, joint_i=0, side=0),
                    legs2body_amplitude, 0
                ])
                # connectivity.append([
                #     bodyosc2index(joint_i=4, side=side_i),
                #     legosc2index(leg_i=1, side_i=side_i, joint_i=0, side=0),
                #     legs2body_amplitude, np.pi
                # ])
                connectivity.append([
                    bodyosc2index(joint_i=i+4, side=side_i),
                    legosc2index(leg_i=1, side_i=side_i, joint_i=0, side=1),
                    legs2body_amplitude, np.pi
                ])
                # connectivity.append([
                #     bodyosc2index(joint_i=4, side=side_i),
                #     legosc2index(leg_i=1, side_i=side_i, joint_i=0, side=1),
                #     legs2body_amplitude, 0
                # ])
        return connectivity

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
                bodyosc2index(joint_i=i, side=1),
                bodyosc2index(joint_i=i, side=0),
                body_amplitude, np.pi
            ])
            connectivity.append([
                bodyosc2index(joint_i=i, side=0),
                bodyosc2index(joint_i=i, side=1),
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
                    bodyosc2index(joint_i=i+1, side=side),
                    bodyosc2index(joint_i=i, side=side),
                    body_amplitude, phase_diff
                ])
                connectivity.append([
                    bodyosc2index(joint_i=i, side=side),
                    bodyosc2index(joint_i=i+1, side=side),
                    body_amplitude, phase_diff
                ])
        # i+1 - i+1 (final)
        connectivity.append([
            bodyosc2index(joint_i=n_body_joints-1, side=1),
            bodyosc2index(joint_i=n_body_joints-1, side=0),
            body_amplitude, np.pi
        ])
        connectivity.append([
            bodyosc2index(joint_i=n_body_joints-1, side=0),
            bodyosc2index(joint_i=n_body_joints-1, side=1),
            body_amplitude, np.pi
        ])

        # Legs (internal)
        for leg_i in range(2):
            for side_i in range(2):
                # 0 - 0
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legs_amplitude, np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legs_amplitude, np.pi
                ])
                # 0 - 1
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legs_amplitude, 0.5*np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legs_amplitude, -0.5*np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legs_amplitude, 0.5*np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legs_amplitude, -0.5*np.pi
                ])
                # 1 - 1
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legs_amplitude, np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legs_amplitude, np.pi
                ])
                # 0 - 2
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legs_amplitude, 0
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    legs_amplitude, 0
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legs_amplitude, 0
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    legs_amplitude, 0
                ])
                # 2 - 2
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    legs_amplitude, np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    legs_amplitude, np.pi
                ])
                # 1 - 3
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legs_amplitude, 0
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=0),
                    legs_amplitude, 0
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legs_amplitude, 0
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=1),
                    legs_amplitude, 0
                ])
                # 3 - 3
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=0),
                    legs_amplitude, np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=1),
                    legs_amplitude, np.pi
                ])

        # Opposite leg interaction
        # TODO
        for leg_i in range(2):
            for side in range(2):
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=0, joint_i=0, side=side),
                    legosc2index(leg_i=leg_i, side_i=1, joint_i=0, side=side),
                    legs_amplitude, np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=1, joint_i=0, side=side),
                    legosc2index(leg_i=leg_i, side_i=0, joint_i=0, side=side),
                    legs_amplitude, np.pi
                ])

        # Following leg interaction
        # TODO
        for side_i in range(2):
            for side in range(2):
                connectivity.append([
                    legosc2index(leg_i=0, side_i=side_i, joint_i=0, side=side),
                    legosc2index(leg_i=1, side_i=side_i, joint_i=0, side=side),
                    legs_amplitude, np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=1, side_i=side_i, joint_i=0, side=side),
                    legosc2index(leg_i=0, side_i=side_i, joint_i=0, side=side),
                    legs_amplitude, np.pi
                ])

        # Body-legs interaction
        for side_i in range(2):
            for i in [0, 1, 7, 8, 9, 10]:
                # Forelimbs
                connectivity.append([
                    bodyosc2index(joint_i=i, side=side_i),
                    legosc2index(leg_i=0, side_i=side_i, joint_i=0, side=0),
                    legs2body_amplitude, 0
                ])
                # connectivity.append([
                #     bodyosc2index(joint_i=1, side=side_i),
                #     legosc2index(leg_i=0, side_i=side_i, joint_i=0, side=0),
                #     legs2body_amplitude, np.pi
                # ])
                connectivity.append([
                    bodyosc2index(joint_i=i, side=side_i),
                    legosc2index(leg_i=0, side_i=side_i, joint_i=0, side=1),
                    legs2body_amplitude, np.pi
                ])
                # connectivity.append([
                #     bodyosc2index(joint_i=1, side=side_i),
                #     legosc2index(leg_i=0, side_i=side_i, joint_i=0, side=1),
                #     legs2body_amplitude, 0
                # ])
            for i in [2, 3, 4, 5]:
                # Hind limbs
                connectivity.append([
                    bodyosc2index(joint_i=i+4, side=side_i),
                    legosc2index(leg_i=1, side_i=side_i, joint_i=0, side=0),
                    legs2body_amplitude, 0
                ])
                # connectivity.append([
                #     bodyosc2index(joint_i=4, side=side_i),
                #     legosc2index(leg_i=1, side_i=side_i, joint_i=0, side=0),
                #     legs2body_amplitude, np.pi
                # ])
                connectivity.append([
                    bodyosc2index(joint_i=i+4, side=side_i),
                    legosc2index(leg_i=1, side_i=side_i, joint_i=0, side=1),
                    legs2body_amplitude, np.pi
                ])
                # connectivity.append([
                #     bodyosc2index(joint_i=4, side=side_i),
                #     legosc2index(leg_i=1, side_i=side_i, joint_i=0, side=1),
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
                bodyosc2index(joint_i=i, side=1),
                bodyosc2index(joint_i=i, side=0),
                body_amplitude, np.pi
            ])
            connectivity.append([
                bodyosc2index(joint_i=i, side=0),
                bodyosc2index(joint_i=i, side=1),
                body_amplitude, np.pi
            ])
            # i - i+1
            for side in range(2):
                connectivity.append([
                    bodyosc2index(joint_i=i+1, side=side),
                    bodyosc2index(joint_i=i, side=side),
                    body_amplitude, 2*np.pi/n_body_joints
                ])
                connectivity.append([
                    bodyosc2index(joint_i=i, side=side),
                    bodyosc2index(joint_i=i+1, side=side),
                    body_amplitude, -2*np.pi/n_body_joints
                ])
        # i+1 - i+1 (final)
        connectivity.append([
            bodyosc2index(joint_i=n_body_joints-1, side=1),
            bodyosc2index(joint_i=n_body_joints-1, side=0),
            body_amplitude, np.pi
        ])
        connectivity.append([
            bodyosc2index(joint_i=n_body_joints-1, side=0),
            bodyosc2index(joint_i=n_body_joints-1, side=1),
            body_amplitude, np.pi
        ])

        # Legs (internal)
        for leg_i in range(2):
            for side_i in range(2):
                # 0 - 0
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legs_amplitude, np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legs_amplitude, np.pi
                ])
                # 0 - 1
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legs_amplitude, 0.5*np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legs_amplitude, -0.5*np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legs_amplitude, 0.5*np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legs_amplitude, -0.5*np.pi
                ])
                # 1 - 1
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legs_amplitude, np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legs_amplitude, np.pi
                ])
                # 1 - 2
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legs_amplitude, 0
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=0),
                    legs_amplitude, 0
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legs_amplitude, 0
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=1),
                    legs_amplitude, 0
                ])
                # 2 - 2
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=0),
                    legs_amplitude, 0
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=1),
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
                legosc2index(leg_i=0, side_i=side_i, joint_i=0, side=0),
                bodyosc2index(joint_i=1, side=side_i),
                legs2body_amplitude, np.pi
            ])
            connectivity.append([
                bodyosc2index(joint_i=1, side=side_i),
                legosc2index(leg_i=0, side_i=side_i, joint_i=0, side=0),
                legs2body_amplitude, np.pi
            ])
            connectivity.append([
                legosc2index(leg_i=0, side_i=side_i, joint_i=0, side=1),
                bodyosc2index(joint_i=1, side=side_i),
                legs2body_amplitude, 0
            ])
            connectivity.append([
                bodyosc2index(joint_i=1, side=side_i),
                legosc2index(leg_i=0, side_i=side_i, joint_i=0, side=1),
                legs2body_amplitude, 0
            ])
            # Hind limbs
            connectivity.append([
                legosc2index(leg_i=1, side_i=side_i, joint_i=0, side=0),
                bodyosc2index(joint_i=4, side=side_i),
                legs2body_amplitude, np.pi
            ])
            connectivity.append([
                bodyosc2index(joint_i=4, side=side_i),
                legosc2index(leg_i=1, side_i=side_i, joint_i=0, side=0),
                legs2body_amplitude, np.pi
            ])
            connectivity.append([
                legosc2index(leg_i=1, side_i=side_i, joint_i=0, side=1),
                bodyosc2index(joint_i=4, side=side_i),
                legs2body_amplitude, 0
            ])
            connectivity.append([
                bodyosc2index(joint_i=4, side=side_i),
                legosc2index(leg_i=1, side_i=side_i, joint_i=0, side=1),
                legs2body_amplitude, 0
            ])
        return connectivity

    @classmethod
    def default(cls):
        """Parameters for walking"""
        connectivity = cls.walking_parameters()
        return cls(np.array(connectivity))

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


class SalamanderJointsArray(JointsArray):
    """Oscillator array"""

    @staticmethod
    def default_parameters():
        """Walking parameters"""
        n_body = 11
        n_dof_legs = 4
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
    def walking_parameters():
        """Walking parameters"""
        n_body = 11
        n_dof_legs = 4
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
        n_dof_legs = 4
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
    def default(cls):
        """Parameters for walking"""
        offsets, rates = cls.default_parameters()
        return cls.from_parameters(offsets, rates)

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


class SalamanderNetworkODE(ODESolver):
    """Salamander network"""

    def __init__(self, state, parameters, timestep):
        super(SalamanderNetworkODE, self).__init__(
            ode=ODE(
                solver=rk4,
                function=ode_oscillators_sparse,
                gradient=ode_oscillators_sparse_gradient
            ),
            state=state.array,
            timestep=timestep,
            parameters=parameters.to_ode_parameters()
        )
        self.state = state
        self.parameters = parameters
        self._n_oscillators = state.n_oscillators
        self._n_joints = parameters.joints.shape()[1]
        n_body = 11
        n_legs_dofs = 4
        # n_legs = 4
        self.group0 = [
            bodyosc2index(joint_i=i, side=0)
            for i in range(11)
        ] + [
            legosc2index(leg_i=leg_i, side_i=side_i, joint_i=joint_i, side=0)
            for leg_i in range(2)
            for side_i in range(2)
            for joint_i in range(n_legs_dofs)
        ]
        self.group1 = [
            bodyosc2index(joint_i=i, side=1)
            for i in range(n_body)
        ] + [
            legosc2index(leg_i=leg_i, side_i=side_i, joint_i=joint_i, side=1)
            for leg_i in range(2)
            for side_i in range(2)
            for joint_i in range(n_legs_dofs)
        ]

        # Adaptive timestep
        self.n_states = len(self.state.array[0, 0, :])
        self.dstate = np.zeros([self.n_states], dtype=np.float64)
        self._jac = np.zeros([self.n_states, self.n_states], dtype=np.float64)
        self.solver = integrate.ode(f=self.fun)  # , jac=self.jac
        self.solver.set_integrator("dopri5")
        self._time = 0
        self._parameters = self.parameters.to_ode_parameters().function

    @classmethod
    def default(cls, n_iterations, timestep):
        """Salamander swimming network"""
        state = SalamanderOscillatorNetworkState.default_state(n_iterations)
        parameters = SalamanderNetworkParameters.default()
        return cls(state, parameters, timestep)

    def fun(self, _time, state):
        """ODE function"""
        self.ode.function(
            self.dstate,
            state,
            *self._parameters
        )
        return self.dstate

    def jac(self, _time, state):
        """ODE function"""
        # self._jac = np.zeros([self.n_states, self.n_states], dtype=np.float64)
        self.ode.gradient(
            self._jac,
            state,
            *self._parameters
        )
        # np.set_printoptions(precision=3, linewidth=np.inf, threshold=np.inf)
        # print(self._jac)
        # raise Exception
        return self._jac

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
        self._parameters = self.parameters.to_ode_parameters().function

    @classmethod
    def walking(cls, n_iterations, timestep):
        """Salamander swimming network"""
        state = SalamanderOscillatorNetworkState.default_state(n_iterations)
        parameters = SalamanderNetworkParameters.for_walking()
        return cls(state, parameters, timestep)

    @classmethod
    def swimming(cls, n_iterations, timestep):
        """Salamander swimming network"""
        state = SalamanderOscillatorNetworkState.default_state(n_iterations)
        parameters = SalamanderNetworkParameters.for_swimming()
        return cls(state, parameters, timestep)

    def control_step(self):
        """Control step"""
        # # Fixed timestep
        # self.step()

        # Adaptive timestep (ODE)
        self.solver.set_initial_value(
            self.state.array[self._iteration, 0, :],
            self._time
        )
        self._time += self._timestep
        self.state.array[self._iteration+1, 0, :] = (
            self.solver.integrate(self._time)
        )
        self.state.array[self._iteration+1, 1, :] = (
            self.state.array[self._iteration+1, 0, :]
            - self.state.array[self._iteration, 0, :]
        )/self._timestep
        self._iteration += 1

        # # Adaptive timestep (ODEINT)
        # self.state.array[self._iteration+1, 0, :] = integrate.odeint(
        #     func=self.fun,
        #     Dfun=self.jac,
        #     y0=np.copy(self.state.array[self._iteration, 0, :]),
        #     t=np.linspace(0, self._timestep, 10),
        #     tfirst=True
        # )[-1]
        # self._iteration += 1

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
        self.parameters.joints.update_drives(drive_speed, drive_turn)
