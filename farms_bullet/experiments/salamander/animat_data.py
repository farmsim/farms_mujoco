"""Animat data"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from .animat_options import SalamanderOptions
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
    HydrodynamicsArray
)


class SalamanderOscillatorNetworkState(OscillatorNetworkState):
    """Network state"""

    @staticmethod
    def default_initial_state():
        """Default state"""
        n_dof_legs = 4
        n_joints = 11 + 4 * n_dof_legs
        return 1e-3 * np.arange(5 * n_joints) + np.concatenate([
            np.linspace(2 * np.pi, 0, n_joints),
            np.zeros(n_joints),
            np.zeros(2 * n_joints),
            np.zeros(n_joints)
        ])

    @staticmethod
    def default_state(n_iterations):
        """Default state"""
        n_dof_legs = 4
        n_joints = 11 + 4 * n_dof_legs
        n_oscillators = 2 * n_joints
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
        hydrodynamics = SalamanderHydrodynamicsArray.from_options(
            options,
            n_iterations
        )
        sensors = Sensors(contacts, proprioception, hydrodynamics)
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
        n_oscillators = 2 * (options.morphology.n_joints_body)
        n_oscillators = 2 * (options.morphology.n_joints())
        freqs_body = 2 * np.pi * np.ones(2 * options.morphology.n_joints_body) * (
            osc_options.body_freqs.value(drives)
        )
        freqs_legs = 2 * np.pi * np.ones(2 * options.morphology.n_joints_legs()) * (
            osc_options.legs_freqs.value(drives)
        )
        freqs = np.concatenate([freqs_body, freqs_legs])
        rates = 10 * np.ones(n_oscillators)
        # Amplitudes
        amplitudes = np.zeros(n_oscillators)
        for i in range(n_body):
            amplitudes[[i, i + n_body]] = 0.1 + 0.2 * i / (n_body - 1)
            # osc_options.body_stand_amplitude*np.sin(
            #     2*np.pi*i/n_body
            #     - osc_options.body_stand_shift
            # )
        for leg_i in range(n_legs):
            for i in range(n_dof_legs):
                amplitudes[[
                    2 * n_body + 2 * leg_i * n_dof_legs + i,
                    2 * n_body + 2 * leg_i * n_dof_legs + i + n_dof_legs
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
            osc_options.body_stand_amplitude * np.sin(
                2 * np.pi * i / n_body_joints
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
        for i in range(n_body_joints - 1):
            # i - i+1
            phase_diff = (
                    2 * np.pi / 11
                # if np.sign(amplitudes[i]) == np.sign(amplitudes[i+1])
                # else np.pi/11+np.pi
            )
            # phase_diff = np.pi/11
            for side in range(2):
                connectivity.append([
                    bodyosc2index(joint_i=i + 1, side=side),
                    bodyosc2index(joint_i=i, side=side),
                    body_amplitude, phase_diff
                ])
                connectivity.append([
                    bodyosc2index(joint_i=i, side=side),
                    bodyosc2index(joint_i=i + 1, side=side),
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
                    legs_amplitude, 0.5 * np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                    legs_amplitude, -0.5 * np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legs_amplitude, 0.5 * np.pi
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                    legs_amplitude, -0.5 * np.pi
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
                    for side_leg in range(2):  # Muscle facing front/back
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
                                    side=(side_i + lateral) % 2
                                ),
                                legosc2index(
                                    leg_i=leg_i,
                                    side_i=side_i,
                                    joint_i=0,
                                    side=(side_i + side_leg) % 2
                                ),
                                legs2body_amplitude,
                                (
                                        walk_phase
                                        + np.pi * (side_i + 1)
                                        + lateral * np.pi
                                        + side_leg * np.pi
                                        + leg_i * np.pi
                                )
                            ])
        return connectivity

    @staticmethod
    def show_connectivity():
        n_joints = 11
        n_dofs_leg = 4
        n_leg = 4
        dim_body = n_joints * 2
        dim = 2 * n_joints + 2 * n_leg * n_dofs_leg
        options = SalamanderOptions()
        C = SalamanderOscillatorConnectivityArray.set_options(options)
        contact_array = SalamanderContactsConnectivityArray.export_params()
        contact_array = np.array(contact_array)
        array = np.array(C)
        G = nx.DiGraph()
        plt.figure()
        pos = np.zeros([dim, 2])
        scale_factor = 0.5
        offset_leg = 1.5

        for i in np.arange(dim):

            if i < dim_body:
                G.add_node(i, pos=(-scale_factor, -scale_factor * (i)))
                if i >= n_joints:
                    G.add_node(i, pos=(scale_factor, -scale_factor * (i - n_joints)))
            if i < dim_body + n_dofs_leg and i >= dim_body:
                G.add_node(i, pos=(scale_factor * (-i + dim_body) - offset_leg, 0))
            if i < dim_body + 2 * n_dofs_leg and i >= dim_body + n_dofs_leg:
                G.add_node(i, pos=(scale_factor * (-i + dim_body + n_dofs_leg) - offset_leg, -scale_factor))
            if i < dim_body + 3 * n_dofs_leg and i >= dim_body + 2 * n_dofs_leg:
                G.add_node(i, pos=(scale_factor * (i - dim_body - 2 * n_dofs_leg) + offset_leg, 0))
            if i < dim_body + 4 * n_dofs_leg and i >= dim_body + 3 * n_dofs_leg:
                G.add_node(i, pos=(scale_factor * (i - dim_body - 3 * n_dofs_leg) + offset_leg, -scale_factor))
            if i < dim_body + 5 * n_dofs_leg and i >= dim_body + 4 * n_dofs_leg:
                G.add_node(i, pos=(scale_factor * (-i + dim_body + 4 * n_dofs_leg) - offset_leg, -3))
            if i < dim_body + 6 * n_dofs_leg and i >= dim_body + 5 * n_dofs_leg:
                G.add_node(i, pos=(scale_factor * (-i + dim_body + 5 * n_dofs_leg) - offset_leg, -3.5))
            if i < dim_body + 7 * n_dofs_leg and i >= dim_body + 6 * n_dofs_leg:
                G.add_node(i, pos=(scale_factor * (i - dim_body - 6 * n_dofs_leg) + offset_leg, -3))
            if i < dim_body + 8 * n_dofs_leg and i >= dim_body + 7 * n_dofs_leg:
                G.add_node(i, pos=(scale_factor * (i - dim_body - 7 * n_dofs_leg) + offset_leg, -3.5))

        G.add_node(dim + 1, pos=(-5, -1), node_color='r')
        G.add_node(dim + 2, pos=(5, -1))
        G.add_node(dim + 3, pos=(-5, -2))
        G.add_node(dim + 4, pos=(5, -2))
        G.add_weighted_edges_from(array[:, 0:3], colors='k')

        G.add_weighted_edges_from(
            np.vstack((contact_array[:, 0], contact_array[:, 1] + 55, np.zeros(len(contact_array)))).T,
            colors='r')
        graph_pose = nx.get_node_attributes(G, 'pos')
        M = G.reverse()
        colors = ['g'] * dim + ['r'] * 4
        nx.draw(M, with_labels=True, node_color=colors, node_size=500, pos=graph_pose)
        plt.axis('equal')
        plt.show()
        return

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
        n_joints = n_body + n_legs * n_dof_legs
        offsets = np.zeros(n_joints)
        for leg_i in range(n_legs):
            for i in range(n_dof_legs):
                offsets[n_body + leg_i * n_dof_legs + i] = (
                    j_options.legs_joints_offsets[i].value(
                        options.control.drives
                    )
                )

        max_amp = 0.2
        low_sat = 1
        up_sat = 3
        if low_sat <= options.control.drives.left <= up_sat:
            if low_sat <= options.control.drives.left <= low_sat + max_amp:
                offsets[0:11] = (-options.control.drives.left + low_sat)
                offsets[19] = -options.control.drives.left
                offsets[23] = +options.control.drives.left
                offsets[11] = +options.control.drives.left
                offsets[15] = -options.control.drives.left
            else:
                offsets[0:11] = -max_amp
                offsets[19] = -max_amp
                offsets[23] = +max_amp
                offsets[11] = +max_amp
                offsets[15] = -max_amp

        if low_sat <= options.control.drives.right <= up_sat:
            if low_sat <= options.control.drives.right <= low_sat + max_amp:
                offsets[0:11] = (options.control.drives.right-low_sat)
                offsets[19] = options.control.drives.right
                offsets[23] = -options.control.drives.right
                offsets[11] = -options.control.drives.right
                offsets[15] = options.control.drives.right
            else:
                offsets[0:11] = max_amp
                offsets[19] = max_amp
                offsets[23] = -max_amp
                offsets[11] = -max_amp
                offsets[15] = max_amp

        rates = 5 * np.ones(n_joints)
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
        foot_id = [26, 30, 34, 38, 42, 46, 50, 54]
        sigma = -0.1
        # connection for foot
        for leg_i in range(2):
            for side_i in range(2):
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=0),
                    2 * leg_i + side_i,
                    sigma
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=1),
                    2 * leg_i + side_i,
                    sigma
                ])

        return cls(np.array(connectivity, dtype=np.float64))

    @staticmethod
    def export_params():
        connectivity = []
        foot_id = [26, 30, 34, 38, 42, 46, 50, 54]
        sigma_foot = -0.1
        sigma_shoulder = -2
        # connection for foot
        for leg_i in range(2):
            for side_i in range(2):
                # ====================foot=======================================
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=0),
                    2 * leg_i + side_i,
                    sigma_foot
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=3, side=1),
                    2 * leg_i + side_i,
                    sigma_foot
                ])
                # ====================lower shoulder============================
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                    2 * leg_i + side_i,
                    sigma_shoulder
                ])
                connectivity.append([
                    legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                    2 * leg_i + side_i,
                    sigma_shoulder
                ])

        print(np.array(connectivity))
        return connectivity


class SalamanderProprioceptionArray(ProprioceptionArray):
    """Salamander proprioception sensors array"""

    @classmethod
    def from_options(cls, options, n_iterations):
        """Default"""
        n_joints = options.morphology.n_joints()
        proprioception = np.zeros([n_iterations, n_joints, 9])
        return cls(proprioception)


class SalamanderHydrodynamicsArray(HydrodynamicsArray):
    """Salamander hydrodynamics sensors array"""

    @classmethod
    def from_options(cls, options, n_iterations):
        """Default"""
        n_body = options.morphology.n_links_body()
        hydrodynamics = np.zeros([n_iterations, n_body, 6])  # Fxyz, Mxyz
        return cls(hydrodynamics)
