"""Animat options"""

import numpy as np
from scipy import interpolate
import pdb

from ...simulations.simulation_options import Options


class AmphibiousOptions(Options):
    """Simulation options"""

    def __init__(self, **kwargs):
        super(AmphibiousOptions, self).__init__()
        self.morphology = kwargs.pop(
            "morphology",
            AmphibiousMorphologyOptions(kwargs)
        )
        self.control = kwargs.pop(
            "control",
            AmphibiousControlOptions(self.morphology, **kwargs)
        )
        self.collect_gps = kwargs.pop(
            "collect_gps",
            False
        )
        self.show_hydrodynamics = kwargs.pop(
            "show_hydrodynamics",
            False
        )
        self.transition = kwargs.pop(
            "transition",
            False
        )
        if kwargs:
            raise Exception("Unknown kwargs: {}".format(kwargs))


class AmphibiousMorphologyOptions(Options):
    """Amphibious morphology options"""

    def __init__(self, options):
        super(AmphibiousMorphologyOptions, self).__init__()
        self.mesh_directory = ""
        self.scale = options.pop("scale", 1.0)
        self.n_joints_body = options.pop("n_joints_body", 11)
        self.n_dof_legs = options.pop("n_dof_legs", 4)
        self.legs_parents = options.pop("legs_parents", [0, 4])
        assert len(self.legs_parents) == self.n_dof_legs//2
        self.n_legs = options.pop("n_legs", 4)
        self.leg_offset = options.pop("leg_offset", 0.03)
        self.leg_length = options.pop("leg_length", 0.06)
        self.leg_radius = options.pop("leg_radius", 0.015)

    def n_joints(self):
        """Number of joints"""
        return self.n_joints_body + self.n_legs*self.n_dof_legs

    def n_joints_legs(self):
        """Number of legs joints"""
        return self.n_legs*self.n_dof_legs

    def n_links_body(self):
        """Number of body links"""
        return self.n_joints_body + 1

    def n_links(self):
        """Number of links"""
        return self.n_links_body() + self.n_joints_legs()


class AmphibiousControlOptions(Options):
    """Amphibious control options"""

    def __init__(self, morphology, **kwargs):
        super(AmphibiousControlOptions, self).__init__()
        self.drives = kwargs.pop(
            "drives",
            AmphibiousDrives(**kwargs)
        )
        self.joints_controllers = kwargs.pop(
            "joints_controllers",
            AmphibiousJointsControllers(**kwargs)
        )
        self.network = kwargs.pop(
            "network",
            AmphibiousNetworkOptions(morphology, **kwargs)
        )

    def to_vector(self):
        """To vector"""
        return [
            self["frequency"],
            self["body_amplitude_0"],
            self["body_amplitude_1"],
            self["body_stand_amplitude"],
            self["body_stand_shift"],
            self["leg_0_amplitude"],
            self["leg_0_offset"],
            self["leg_1_amplitude"],
            self["leg_1_offset"],
            self["leg_2_amplitude"],
            self["leg_2_offset"],
            self["leg_turn"],
            self["body_p"],
            self["body_d"],
            self["body_f"],
            self["legs_p"],
            self["legs_d"],
            self["legs_f"]
        ]

    def from_vector(self, vector):
        """From vector"""
        (
            self["frequency"],
            self["body_amplitude_0"],
            self["body_amplitude_1"],
            self["body_stand_amplitude"],
            self["body_stand_shift"],
            self["leg_0_amplitude"],
            self["leg_0_offset"],
            self["leg_1_amplitude"],
            self["leg_1_offset"],
            self["leg_2_amplitude"],
            self["leg_2_offset"],
            self["leg_turn"],
            self["body_p"],
            self["body_d"],
            self["body_f"],
            self["legs_p"],
            self["legs_d"],
            self["legs_f"]
        ) = vector


class AmphibiousDrives(Options):
    """Amphibious drives"""

    def __init__(self, **kwargs):
        super(AmphibiousDrives, self).__init__()
        self.forward = kwargs.pop("drive_forward", 2)
        self.turning = kwargs.pop("drive_turn", 0)
        self.left = kwargs.pop("drive_left", 0)
        self.right = kwargs.pop("drive_right", 0)


class AmphibiousJointsControllers(Options):
    """Amphibious joints controllers"""

    def __init__(self, **kwargs):
        super(AmphibiousJointsControllers, self).__init__()
        self.body_p = kwargs.pop("body_p", 1e-1)
        self.body_d = kwargs.pop("body_d", 1e0)
        self.body_f = kwargs.pop("body_f", 1e1)
        self.legs_p = kwargs.pop("legs_p", 1e-1)
        self.legs_d = kwargs.pop("legs_d", 1e0)
        self.legs_f = kwargs.pop("legs_f", 1e1)


class AmphibiousNetworkOptions(Options):
    """Amphibious network options"""

    def __init__(self, morphology, **kwargs):
        super(AmphibiousNetworkOptions, self).__init__()
        self.oscillators = kwargs.pop(
            "oscillators",
            AmphibiousOscillatorOptions(morphology, **kwargs)
        )
        self.connectivity = kwargs.pop(
            "connectivity",
            AmphibiousConnectivityOptions(morphology, **kwargs)
        )
        self.joints = kwargs.pop(
            "joints",
            AmphibiousJointsOptions(morphology, **kwargs)
        )
        self.sensors = kwargs.pop(
            "sensors",
            None
        )

    def update(self):
        """Update"""
        self.oscillators.update()
        self.joints.update()


class DriveDependentProperty(Options):
    """Drive dependent property"""

    def __init__(self, data):
        super(DriveDependentProperty, self).__init__()
        _data = np.array(data)
        self.interp = interpolate.interp1d(_data[:, 0], _data[:, 1])

    # def forward(self, drives):
    #     return self.interp(drives.forward)

    def value(self, drives):
        """Value in function of drive"""
        return self.interp(drives.forward)


class AmphibiousOscillatorFrequenciesOptions(DriveDependentProperty):
    """Amphibious oscillator frequencies options"""

    @classmethod
    def legs_freqs(cls):
        """Legs intrinsic frequencies"""
        return  cls([
            [0, 0],
            [1, 0],
            [1, 0.5],
            [3, 1.5],
            [3, 0],
            [6, 0]
        ])

    @classmethod
    def body_freqs(cls):
        """Body intrinsic frequencies"""
        return cls([
            [0, 0],
            [1, 0],
            [1, 1.5],
            [5, 4],
            [5, 0],
            [6, 0]
        ])

    def value(self, drives):
        """Value in function of drive"""
        return self.interp(drives.forward)


class AmphibiousOscillatorAmplitudeOptions(DriveDependentProperty):
    """Amphibious oscillators amplitudes options"""

    @classmethod
    def legs_nominal_amplitudes(cls, joint_i, **kwargs):
        """Legs nominal amplitudes"""
        amplitude = kwargs.pop(
            "leg_{}_amplitude".format(joint_i),
            [0.8, np.pi/32, np.pi/4, np.pi/8][joint_i]
        )
        return cls([
            [0, 0],
            [1, 0],
            [1, 0.7*amplitude],
            [3, amplitude],
            [3, 0],
            [6, 0]
        ])

    @classmethod
    def body_nominal_amplitudes(cls, morphology, joint_i, **kwargs):
        """Body nominal amplitudes"""
        body_stand_amplitude = kwargs.pop("body_stand_amplitude", 0.2)
        n_body = morphology.n_joints_body
        body_stand_shift = np.pi/4
        amplitude = body_stand_amplitude*np.sin(
            2*np.pi*joint_i/n_body - body_stand_shift
        )
        # osc_options.body_stand_amplitude*np.sin(
        #     2*np.pi*i/n_body
        #     - osc_options.body_stand_shift
        # )
        return cls([
            [0, 0.3*amplitude],
            [3, amplitude],
            [3, 0.1*joint_i/n_body],
            [5, 0.6*joint_i/n_body+0.2],
            [5, 0],
            [6, 0]
        ])

    # def show_body_amplitudes(self):
    #     for joint_i in range(11):
    #         print(self.body_nominal_amplitudes(joint_i=joint_i))

    @staticmethod
    def joint_value(options, joint_i):
        """Value in function of drive"""
        n_body = options.morphology.n_joints_body
        osc_options = options.control.network.oscillators
        return osc_options.body_stand_amplitude*np.sin(
            2*np.pi*joint_i/n_body
            - osc_options.body_stand_shift
        )


class AmphibiousOscillatorJointsOptions(DriveDependentProperty):
    """Amphibious drive dependent properties"""

    @classmethod
    def legs_joints_offsets(cls, joint_i, **kwargs):
        """Legs joints offsets"""
        offsets_walking = kwargs.pop(
            "legs_offsets_walking",
            [0, np.pi/32, 0, np.pi/8]
        )
        offsets_swimming = kwargs.pop(
            "legs_offsets_swimming",
            [-2*np.pi/5, 0, 0, 0]
        )
        return cls([
            [0, offsets_swimming[joint_i]],
            [1, offsets_swimming[joint_i]],
            [1, offsets_walking[joint_i]],
            [3, offsets_walking[joint_i]],
            [3, offsets_swimming[joint_i]],
            [6, offsets_swimming[joint_i]]
        ])

    @classmethod
    def body_joints_offsets(cls, joint_i, offset=0):
        """Body joints offsets"""
        return cls([
            [0, offset],
            [6, offset]
        ])


class AmphibiousOscillatorOptions(Options):
    """Amphibious oscillator options

    Includes frequencies, amplitudes rates and nominal amplitudes

    """

    def __init__(self, morphology, **kwargs):
        super(AmphibiousOscillatorOptions, self).__init__()
        self.morphology = morphology
        self.body_head_amplitude = kwargs.pop("body_head_amplitude", 0)
        self.body_tail_amplitude = kwargs.pop("body_tail_amplitude", 0)
        self._body_stand_amplitude = kwargs.pop("body_stand_amplitude", 0.2)
        self._legs_amplitudes = kwargs.pop(
            "legs_amplitude",
            [0.8, np.pi/32, np.pi/4, np.pi/8]
        )
        self._body_stand_shift = kwargs.pop("body_stand_shift", np.pi/4)
        self.body_nominal_amplitudes = None
        self.legs_nominal_amplitudes = None
        self.update()

        # Frequencies
        self.body_freqs = AmphibiousOscillatorFrequenciesOptions.body_freqs()
        self.legs_freqs = AmphibiousOscillatorFrequenciesOptions.legs_freqs()

    def update(self):
        """Update all"""
        self.set_body_nominal_amplitudes()
        self.set_legs_nominal_amplitudes()

    def get_body_stand_amplitude(self):
        """Body stand amplitude"""
        return self._body_stand_amplitude

    def set_body_stand_amplitude(self, value):
        """Body stand amplitude"""
        self._body_stand_amplitude = value
        self.set_body_nominal_amplitudes()

    def set_body_stand_shift(self, value):
        """Body stand shift"""
        self._body_stand_shift = value
        self.set_body_nominal_amplitudes()

    def set_body_nominal_amplitudes(self):
        """Set body nominal amplitudes"""
        self.body_nominal_amplitudes = [
            AmphibiousOscillatorAmplitudeOptions.body_nominal_amplitudes(
                self.morphology,
                joint_i,
                body_stand_amplitude=self._body_stand_amplitude
            )
            for joint_i in range(self.morphology.n_joints_body)
        ]

    def get_legs_amplitudes(self):
        """Body legs amplitude"""
        return self._legs_amplitudes

    def set_legs_amplitudes(self, values):
        """Body legs amplitude"""
        self._legs_amplitudes = values
        self.set_legs_nominal_amplitudes()

    def set_legs_nominal_amplitudes(self):
        """Set legs nominal amplitudes"""
        self.legs_nominal_amplitudes = [
            AmphibiousOscillatorAmplitudeOptions.legs_nominal_amplitudes(
                joint_i,
                **{
                    "leg_{}_amplitude".format(joint_i): (
                        self._legs_amplitudes[joint_i]
                    )
                }
            )
            for joint_i in range(self.morphology.n_dof_legs)
        ]


class AmphibiousConnectivityOptions(Options):
    """Amphibious connectivity options"""

    def __init__(self, morphology, **kwargs):
        super(AmphibiousConnectivityOptions, self).__init__()
        self.body_phase_bias = kwargs.pop(
            "body_phase_bias",
            2*np.pi/morphology.n_joints_body
        )
        self.leg_phase_follow = kwargs.pop(
            "leg_phase_follow",
            np.pi
        )
        self.weight_osc_body = 1e3
        self.weight_osc_legs_internal = 1e3
        self.weight_osc_legs_opposite = 1e0
        self.weight_osc_legs_following = 1e0
        self.weight_osc_legs2body = 3e1
        self.weight_sens_contact_i = -2e0
        self.weight_sens_contact_e = 2e0  # +3e-1
        self.weight_sens_hydro_freq = 1
        self.weight_sens_hydro_amp = 1


class AmphibiousJointsOptions(Options):
    """Amphibious joints options"""

    def __init__(self, morphology, **kwargs):
        super(AmphibiousJointsOptions, self).__init__()
        self.morphology = morphology
        self._legs_offsets = kwargs.pop(
            "legs_offsets_walking",
            [0, np.pi/32, 0, np.pi/8]
        )
        self._legs_offsets_swimming = kwargs.pop(
            "legs_offsets_swimming",
            [-2*np.pi/5, 0, 0, 0]
        )
        # Joints offsets
        self.legs_offsets = None
        self.update_legs_offsets()
        self._body_offset = 0
        self.body_offsets = None
        self.update_body_offsets()

    def update(self):
        """Update"""
        self.update_body_offsets()
        self.update_legs_offsets()

    def get_legs_offsets(self):
        """Get legs offsets"""
        return self._legs_offsets

    def set_legs_offsets(self, values):
        """Set legs offsets"""
        self._legs_offsets = values
        self.update_legs_offsets()

    def update_legs_offsets(self):
        """Set legs joints offsets"""
        self.legs_offsets = [
            AmphibiousOscillatorJointsOptions.legs_joints_offsets(
                joint_i,
                legs_offsets_walking=self._legs_offsets,
                legs_offsets_swimming=self._legs_offsets_swimming
            )
            for joint_i in range(self.morphology.n_dof_legs)
        ]

    def set_body_offsets(self, value):
        """Set body offsets"""
        self._body_offset = value
        self.update_body_offsets()

    def update_body_offsets(self):
        """Set body joints offsets"""
        self.body_offsets = [
            AmphibiousOscillatorJointsOptions.body_joints_offsets(
                joint_i,
                offset=self._body_offset
            )
            for joint_i in range(self.morphology.n_joints_body)
        ]
