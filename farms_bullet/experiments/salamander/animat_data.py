"""Animat data"""

import numpy as np

from .convention import bodyosc2index, legosc2index
from ...animats.animat_data import (
    OscillatorNetworkState,
    AnimatData,
    NetworkParameters,
    OscillatorArray,
    ConnectivityArray,
    JointsArray,
    Sensors,
    ContactsArray,
    ProprioceptionArray,
    GpsArray,
    HydrodynamicsArray
)


class SalamanderOscillatorNetworkState(OscillatorNetworkState):
    """Network state"""

    @staticmethod
    def default_initial_state():
        """Default state"""
        n_dof_legs = 4
        n_joints = 11+4*n_dof_legs
        return 1e-3*np.arange(5*n_joints) + np.concatenate([
            np.linspace(2*np.pi, 0, n_joints),
            np.zeros(n_joints),
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
        state[0, 0, :] = np.array(initial_state)
        return cls(state, n_oscillators)


class SalamanderData(AnimatData):
    """Salamander network parameter"""

    @classmethod
    def from_options(cls, state, options, n_iterations):
        """Default salamander newtwork parameters"""
        oscillators = SalamanderOscillatorArray.from_options(options)
        connectivity = SalamanderOscillatorConnectivityArray.from_options(options)
        contacts_connectivity = SalamanderContactsConnectivityArray.from_options(
            options
        )
        network = NetworkParameters(
            oscillators,
            connectivity,
            contacts_connectivity
        )
        joints = SalamanderJointsArray.from_options(options)
        contacts = SalamanderContactsArray.from_options(options, n_iterations)
        proprioception = SalamanderProprioceptionArray.from_options(
            options,
            n_iterations
        )
        gps = SalamanderGpsArray.from_options(
            options,
            n_iterations
        )
        hydrodynamics = SalamanderHydrodynamicsArray.from_options(
            options,
            n_iterations
        )
        sensors = Sensors(contacts, proprioception, gps, hydrodynamics)
        return cls(state, network, joints, sensors)


class SalamanderOscillatorArray(OscillatorArray):
    """Oscillator array"""

    @staticmethod
    def set_options(options):
        """Walking parameters"""
        osc_options = options.control.network.oscillators
        drives = options.control.drives
        n_body = options.morphology.n_joints_body
        n_dof_legs = options.morphology.n_dof_legs
        n_legs = options.morphology.n_legs
        n_oscillators = 2*(options.morphology.n_joints_body)
        n_oscillators = 2*(options.morphology.n_joints())
        freqs_body = 2*np.pi*np.ones(2*options.morphology.n_joints_body)*(
            osc_options.body_freqs.value(drives)
        )
        freqs_legs = 2*np.pi*np.ones(2*options.morphology.n_joints_legs())*(
            osc_options.legs_freqs.value(drives)
        )
        freqs = np.concatenate([freqs_body, freqs_legs])
        rates = 10*np.ones(n_oscillators)
        # Amplitudes
        amplitudes = np.zeros(n_oscillators)
        for i in range(n_body):
            # amplitudes[[i, i+n_body]] = 0.1+0.2*i/(n_body-1)
            amplitudes[[i, i+n_body]] = (
                osc_options.body_nominal_amplitudes[i].value(drives)
            )
            # osc_options.body_stand_amplitude*np.sin(
            #     2*np.pi*i/n_body
            #     - osc_options.body_stand_shift
            # )
        for leg_i in range(n_legs):
            for i in range(n_dof_legs):
                amplitudes[[
                    2*n_body + 2*leg_i*n_dof_legs + i,
                    2*n_body + 2*leg_i*n_dof_legs + i + n_dof_legs
                ]] = osc_options.legs_nominal_amplitudes[i].value(drives)
        # print("Amplitudes along body: abs({})".format(amplitudes[:11]))
        return np.abs(freqs), np.abs(rates), np.abs(amplitudes)

    @classmethod
    def from_options(cls, options):
        """Default"""
        freqs, rates, amplitudes = cls.set_options(options)
        return cls.from_parameters(freqs, rates, amplitudes)

    def update(self, options):
        """Update from options

        :param options: Animat options

        """
        freqs, _, amplitudes = self.set_options(options)
        self.freqs[:] = freqs
        self.amplitudes_desired[:] = amplitudes


class SalamanderOscillatorConnectivityArray(ConnectivityArray):
    """Connectivity array"""

    @staticmethod
    def set_options(options):
        """Walking parameters"""
        options = options
        n_body_joints = 11
        connectivity = []
        body_amplitude = 1e2
        legs_amplitude = 1e2
        legs2body_amplitude = 1e3

        osc_options = options.control.network.oscillators
        conn_options = options.control.network.connectivity

        # Amplitudes
        amplitudes = [
            osc_options.body_stand_amplitude*np.sin(
                2*np.pi*i/n_body_joints
                - osc_options.body_stand_shift
            )
            for i in range(n_body_joints)
        ]

        # Body
        for i in range(n_body_joints):
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
        for i in range(n_body_joints-1):
            # i - i+1
            phase_diff = (
                2*np.pi/11
                # if np.sign(amplitudes[i]) == np.sign(amplitudes[i+1])
                # else np.pi/11+np.pi
            )
            # phase_diff = np.pi/11
            for side in range(2):
                connectivity.append([
                    bodyosc2index(joint_i=i+1, side=side),
                    bodyosc2index(joint_i=i, side=side),
                    body_amplitude, phase_diff
                ])
                connectivity.append([
                    bodyosc2index(joint_i=i, side=side),
                    bodyosc2index(joint_i=i+1, side=side),
                    body_amplitude, -phase_diff
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
        for leg_i in range(2):
            for side_i in range(2):
                for i in range(11):  # [0, 1, 7, 8, 9, 10]
                    for side_leg in range(2): # Muscle facing front/back
                        for lateral in range(2):
                            walk_phase = (
                                np.pi
                                if i in [0, 1, 7, 8, 9, 10]
                                else 0
                            )
                            # Forelimbs
                            connectivity.append([
                                bodyosc2index(
                                    joint_i=i,
                                    side=(side_i+lateral)%2
                                ),
                                legosc2index(
                                    leg_i=leg_i,
                                    side_i=side_i,
                                    joint_i=0,
                                    side=(side_i+side_leg)%2
                                ),
                                legs2body_amplitude,
                                (
                                    walk_phase
                                    + np.pi*(side_i+1)
                                    + lateral*np.pi
                                    + side_leg*np.pi
                                    + leg_i*np.pi
                                )
                            ])
        return connectivity

    @classmethod
    def from_options(cls, options):
        """Parameters for walking"""
        connectivity = cls.set_options(options)
        return cls(np.array(connectivity))

    def update(self, options):
        """Update from options

        :param options: Animat options

        """


class SalamanderJointsArray(JointsArray):
    """Oscillator array"""

    @staticmethod
    def set_options(options):
        """Walking parameters"""
        j_options = options.control.network.joints
        n_body = options.morphology.n_joints_body
        n_dof_legs = options.morphology.n_dof_legs
        n_legs = options.morphology.n_legs
        n_joints = n_body + n_legs*n_dof_legs
        offsets = np.zeros(n_joints)
        for leg_i in range(n_legs):
            for i in range(n_dof_legs):
                offsets[n_body + leg_i*n_dof_legs + i] = (
                    j_options.legs_joints_offsets[i].value(
                        options.control.drives
                    )
                )
        rates = 5*np.ones(n_joints)
        return offsets, rates

    @classmethod
    def from_options(cls, options):
        """Parameters for walking"""
        offsets, rates = cls.set_options(options)
        return cls.from_parameters(offsets, rates)

    def update(self, options):
        """Update from options

        :param options: Animat options

        """
        offsets, _ = self.set_options(options)
        self.offsets[:] = offsets


class SalamanderContactsArray(ContactsArray):
    """Salamander contacts sensors array"""

    @classmethod
    def from_options(cls, options, n_iterations):
        """Default"""
        # n_body = options.morphology.n_joints_body
        n_contacts = options.morphology.n_legs
        # n_joints = options.morphology.n_joints()
        contacts = np.zeros([n_iterations, n_contacts, 9])  # x, y, z
        return cls(contacts)


class SalamanderContactsConnectivityArray(ConnectivityArray):
    """Salamander contacts connectivity array"""

    @classmethod
    def from_options(cls, options):
        """Default"""
        connectivity = []
        # options.morphology.n_legs
        for leg_i in range(2):
            for side_i in range(2):
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    2*leg_i + side_i,
                    0  # Weight
                ])
        print(np.array(connectivity))
        return cls(np.array(connectivity, dtype=np.float64))


class SalamanderProprioceptionArray(ProprioceptionArray):
    """Salamander proprioception sensors array"""

    @classmethod
    def from_options(cls, options, n_iterations):
        """Default"""
        n_joints = options.morphology.n_joints()
        proprioception = np.zeros([n_iterations, n_joints, 9])
        return cls(proprioception)


class SalamanderGpsArray(GpsArray):
    """Salamander gps sensors array"""

    @classmethod
    def from_options(cls, options, n_iterations):
        """Default"""
        n_links = options.morphology.n_links()
        gps = np.zeros([n_iterations, n_links, 20])
        return cls(gps)


class SalamanderHydrodynamicsArray(HydrodynamicsArray):
    """Salamander hydrodynamics sensors array"""

    @classmethod
    def from_options(cls, options, n_iterations):
        """Default"""
        n_body = options.morphology.n_links_body()
        hydrodynamics = np.zeros([n_iterations, n_body, 6])  # Fxyz, Mxyz
        return cls(hydrodynamics)
