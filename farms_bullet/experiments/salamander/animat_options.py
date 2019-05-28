"""Animat options"""

import numpy as np
from scipy import interpolate


class SalamanderOptions(dict):
    """Simulation options"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, **kwargs):
        super(SalamanderOptions, self).__init__()
        self.morphology = kwargs.pop(
            "morphology",
            SalamanderMorphologyOptions()
        )
        self.control = kwargs.pop(
            "control",
            SalamanderControlOptions()
        )
        if kwargs:
            raise Exception("Unknown kwargs: {}".format(kwargs))


class SalamanderMorphologyOptions(dict):
    """Salamander morphology options"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, **kwargs):
        super(SalamanderMorphologyOptions, self).__init__()
        self.n_joints_body = kwargs.pop("n_joints_body", 11)
        self.n_dof_legs = kwargs.pop("n_dof_legs", 4)
        self.n_legs = kwargs.pop("n_legs", 4)

    def n_joints(self):
        """Number of joints"""
        return self.n_joints_body + self.n_legs*self.n_dof_legs

    def n_joints_legs(self):
        """Number of legs joints"""
        return self.n_legs*self.n_dof_legs


class SalamanderControlOptions(dict):
    """Salamander control options"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, **kwargs):
        super(SalamanderControlOptions, self).__init__()
        self.drives = kwargs.pop(
            "drives",
            SalamanderDrives(**kwargs)
        )
        self.joints_controllers = kwargs.pop(
            "joints_controllers",
            SalamanderJointsControllers(**kwargs)
        )
        self.network = kwargs.pop(
            "network",
            SalamanderNetworkOptions(**kwargs)
        )

    # @classmethod
    # def from_gait(cls, gait, **kwargs):
    #     """Salamander control option from gait"""
    #     return (
    #         cls.walking(frequency=kwargs.pop("frequency", 1), **kwargs)
    #         if gait == "walking"
    #         else cls.swimming(frequency=kwargs.pop("frequency", 2), **kwargs)
    #         if gait == "swimming"
    #         else cls.standing()
    #     )

    # @classmethod
    # def standing(cls, **kwargs):
    #     """Standing options"""
    #     # Options
    #     options = {}

    #     # General
    #     options["n_body_joints"] = 11
    #     options["frequency"] = kwargs.pop("frequency", 0)

    #     # Body
    #     options["body_amplitude_0"] = kwargs.pop("body_amplitude_0", 0)
    #     options["body_amplitude_1"] = kwargs.pop("body_amplitude_1", 0)
    #     options["body_stand_amplitude"] = kwargs.pop("body_stand_amplitude", 0)
    #     options["body_stand_shift"] = kwargs.pop("body_stand_shift", 0)

    #     # Legs
    #     options["leg_0_amplitude"] = kwargs.pop("leg_0_amplitude", 0)
    #     options["leg_0_offset"] = kwargs.pop("leg_0_offset", 0)

    #     options["leg_1_amplitude"] = kwargs.pop("leg_1_amplitude", 0)
    #     options["leg_1_offset"] = kwargs.pop("leg_1_offset", np.pi/16)

    #     options["leg_2_amplitude"] = kwargs.pop("leg_2_amplitude", 0)
    #     options["leg_2_offset"] = kwargs.pop("leg_2_offset", 0)

    #     options["leg_3_amplitude"] = kwargs.pop("leg_3_amplitude", 0)
    #     options["leg_3_offset"] = kwargs.pop("leg_3_offset", np.pi/8)

    #     # Additional walking options
    #     options["leg_turn"] = 0

    #     # Gains
    #     options["body_p"] = 1e-1
    #     options["body_d"] = 1e0
    #     options["body_f"] = 1e1
    #     options["legs_p"] = 1e-1
    #     options["legs_d"] = 1e0
    #     options["legs_f"] = 1e1

    #     # Additional options
    #     options.update(kwargs)
    #     return cls(options)

    # @classmethod
    # def walking(cls, **kwargs):
    #     """Walking options"""
    #     # Options
    #     options = {}

    #     # General
    #     options["n_body_joints"] = 11
    #     options["frequency"] = kwargs.pop("frequency", 1)

    #     # Body
    #     options["body_amplitude_0"] = kwargs.pop("body_amplitude_0", 0)
    #     options["body_amplitude_1"] = kwargs.pop("body_amplitude_1", 0)
    #     options["body_stand_amplitude"] = kwargs.pop(
    #         "body_stand_amplitude",
    #         0.2
    #     )
    #     options["body_stand_shift"] = kwargs.pop("body_stand_shift", np.pi/4)

    #     # Legs
    #     options["leg_0_amplitude"] = kwargs.pop("leg_0_amplitude", 0.8)
    #     options["leg_0_offset"] = kwargs.pop("leg_0_offset", 0)

    #     options["leg_1_amplitude"] = kwargs.pop("leg_1_amplitude", np.pi/32)
    #     options["leg_1_offset"] = kwargs.pop("leg_1_offset", np.pi/32)

    #     options["leg_2_amplitude"] = kwargs.pop("leg_2_amplitude", np.pi/4)
    #     options["leg_2_offset"] = kwargs.pop("leg_2_offset", 0)

    #     options["leg_3_amplitude"] = kwargs.pop("leg_3_amplitude", np.pi/8)
    #     options["leg_3_offset"] = kwargs.pop("leg_3_offset", np.pi/8)

    #     # Additional walking options
    #     options["leg_turn"] = 0

    #     # Gains
    #     options["body_p"] = 1e-1
    #     options["body_d"] = 1e0
    #     options["body_f"] = 1e1
    #     options["legs_p"] = 1e-1
    #     options["legs_d"] = 1e0
    #     options["legs_f"] = 1e1

    #     # Additional options
    #     options.update(kwargs)
    #     return cls(options)

    # @classmethod
    # def swimming(cls, **kwargs):
    #     """Swimming options"""
    #     # Options
    #     options = {}

    #     # General
    #     n_body_joints = 11
    #     options["n_body_joints"] = n_body_joints
    #     options["frequency"] = kwargs.pop("frequency", 2)

    #     # Body
    #     options["body_amplitude_0"] = kwargs.pop("body_amplitude_0", 0.1)
    #     options["body_amplitude_1"] = kwargs.pop("body_amplitude_1", 0.5)
    #     options["body_stand_amplitude"] = kwargs.pop("body_stand_amplitude", 0)
    #     options["body_stand_shift"] = kwargs.pop("body_stand_shift", 0)

    #     # Legs
    #     options["leg_0_amplitude"] = kwargs.pop("leg_0_amplitude", 0)
    #     options["leg_0_offset"] = kwargs.pop("leg_0_offset", -2*np.pi/5)

    #     options["leg_1_amplitude"] = kwargs.pop("leg_1_amplitude", 0)
    #     options["leg_1_offset"] = kwargs.pop("leg_1_offset", 0)

    #     options["leg_2_amplitude"] = kwargs.pop("leg_2_amplitude", 0)
    #     options["leg_2_offset"] = kwargs.pop("leg_2_offset", 0)

    #     options["leg_3_amplitude"] = kwargs.pop("leg_3_amplitude", 0)
    #     options["leg_3_offset"] = kwargs.pop("leg_3_offset", 0)

    #     # Additional walking options
    #     options["leg_turn"] = 0

    #     # Gains
    #     options["body_p"] = 1e-1
    #     options["body_d"] = 1e0
    #     options["body_f"] = 1e1
    #     options["legs_p"] = 1e-1
    #     options["legs_d"] = 1e0
    #     options["legs_f"] = 1e1

    #     # Additional options
    #     options.update(kwargs)
    #     return cls(options)

    # @classmethod
    # def default(cls, **kwargs):
    #     """Walking options"""
    #     # Options
    #     options = {}

    #     # General
    #     options["n_body_joints"] = 11
    #     options["frequency"] = kwargs.pop("frequency", 1)

    #     # Body
    #     options["body_amplitude_0"] = kwargs.pop("body_amplitude_0", 0)
    #     options["body_amplitude_1"] = kwargs.pop("body_amplitude_1", 0)
    #     options["body_stand_amplitude"] = kwargs.pop(
    #         "body_stand_amplitude",
    #         0.2
    #     )
    #     options["body_stand_shift"] = kwargs.pop("body_stand_shift", np.pi/4)

    #     # Legs
    #     options["leg_0_amplitude"] = kwargs.pop("leg_0_amplitude", 0.8)
    #     options["leg_0_offset"] = kwargs.pop("leg_0_offset", 0)

    #     options["leg_1_amplitude"] = kwargs.pop("leg_1_amplitude", np.pi/32)
    #     options["leg_1_offset"] = kwargs.pop("leg_1_offset", np.pi/32)

    #     options["leg_2_amplitude"] = kwargs.pop("leg_2_amplitude", np.pi/4)
    #     options["leg_2_offset"] = kwargs.pop("leg_2_offset", 0)

    #     options["leg_3_amplitude"] = kwargs.pop("leg_3_amplitude", np.pi/8)
    #     options["leg_3_offset"] = kwargs.pop("leg_3_offset", np.pi/8)

    #     # Additional walking options
    #     options["leg_turn"] = 0

    #     # Gains
    #     options["body_p"] = 1e-1
    #     options["body_d"] = 1e0
    #     options["body_f"] = 1e1
    #     options["legs_p"] = 1e-1
    #     options["legs_d"] = 1e0
    #     options["legs_f"] = 1e1

    #     # Additional options
    #     options.update(kwargs)
    #     return cls(options)

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


class SalamanderDrives(dict):
    """Salamander drives"""

    def __init__(self, **kwargs):
        super(SalamanderDrives, self).__init__()
        self.forward = kwargs.pop("drive_forward", 2)
        self.left = kwargs.pop("drive_left", 0)
        self.right = kwargs.pop("drive_right", 0)


class SalamanderJointsControllers(dict):
    """Salamander joints controllers"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, **kwargs):
        super(SalamanderJointsControllers, self).__init__()
        self.body_p = kwargs.pop("body_p", 1e-1)
        self.body_d = kwargs.pop("body_d", 1e0)
        self.body_f = kwargs.pop("body_f", 1e1)
        self.legs_p = kwargs.pop("legs_p", 1e-1)
        self.legs_d = kwargs.pop("legs_d", 1e0)
        self.legs_f = kwargs.pop("legs_f", 1e1)


class SalamanderNetworkOptions(dict):
    """Salamander network options"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, **kwargs):
        super(SalamanderNetworkOptions, self).__init__()
        self.oscillators = kwargs.pop(
            "oscillators",
            SalamanderOscillatorOptions(**kwargs)
        )
        self.connectivity = kwargs.pop(
            "connectivity",
            SalamanderConnectivityOptions(**kwargs)
        )
        self.joints = kwargs.pop(
            "joints",
            SalamanderJointsOptions(**kwargs)
        )
        self.sensors = kwargs.pop(
            "sensors",
            None
        )


class DriveDependentProperty(dict):
    """Drive dependent property"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, data):
        super(DriveDependentProperty, self).__init__()
        _data = np.array(data)
        self.interp = interpolate.interp1d(_data[:, 0], _data[:, 1])

    def value(self, drives):
        """Value in function of drive"""
        return self.interp(drives.forward)


class SalamanderOscillatorFrequenciesOptions(DriveDependentProperty):
    """Salamander oscillator frequencies options"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def legs_freqs(cls):
        """Legs intrinsic frequencies"""
        return  cls([
            [0, 0],
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
            [5, 4],
            [5, 0],
            [6, 0]
        ])

    def value(self, drives):
        """Value in function of drive"""
        return self.interp(drives.forward)


class SalamanderOscillatorAmplitudeOptions(DriveDependentProperty):
    """Salamander oscillators amplitudes options"""

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
    def body_nominal_amplitudes(cls):
        """Body nominal amplitudes"""
        amplitude = 0.2
        # osc_options.body_stand_amplitude*np.sin(
        #         2*np.pi*i/n_body
        #         - osc_options.body_stand_shift
        #     )
        # osc_options.body_stand_amplitude*np.sin(
        #     2*np.pi*i/n_body
        #     - osc_options.body_stand_shift
        # )
        return cls([
            [0, 0.5*amplitude],
            [6, amplitude]
        ])

    @staticmethod
    def joint_value(options, joint_i):
        """Value in function of drive"""
        n_body = options.morphology.n_joints_body
        osc_options = options.control.network.oscillators
        return osc_options.body_stand_amplitude*np.sin(
            2*np.pi*joint_i/n_body
            - osc_options.body_stand_shift
        )


class SalamanderOscillatorJointsOptions(DriveDependentProperty):
    """Salamander drive dependent properties"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def legs_joints_offsets(cls, joint_i, **kwargs):
        """Legs joints offsets"""
        offsets_walking = kwargs.pop(
            "legs_offsets_walking",
            [0, np.pi/32, 0, np.pi/8]
        )
        offsets_swimming = kwargs.pop(
            "legs_offsets_walking",
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
    def body_joints_offsets(cls):
        """Body joints offsets"""
        return cls([
            [0, 0],
            [6, 0]
        ])


class SalamanderOscillatorOptions(dict):
    """Salamander oscillator options

    Includes frequencies, amplitudes rates and nominal amplitudes

    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, **kwargs):
        super(SalamanderOscillatorOptions, self).__init__()

        self.body_head_amplitude = kwargs.pop("body_head_amplitude", 0)
        self.body_tail_amplitude = kwargs.pop("body_tail_amplitude", 0)
        self.body_stand_amplitude = kwargs.pop("body_stand_amplitude", 0.2)
        self.body_stand_shift = kwargs.pop("body_stand_shift", np.pi/4)

        # Frequencies
        self.body_freqs = SalamanderOscillatorFrequenciesOptions.body_freqs()
        self.legs_freqs = SalamanderOscillatorFrequenciesOptions.legs_freqs()

        # Nominal amplitudes
        self.body_nominal_amplitudes = (
            SalamanderOscillatorAmplitudeOptions.body_nominal_amplitudes()
        )
        self.legs_nominal_amplitudes = [
            SalamanderOscillatorAmplitudeOptions.legs_nominal_amplitudes(
                joint_i
            )
            for joint_i in range(4)
        ]


class SalamanderConnectivityOptions(dict):
    """Salamander connectivity options"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, **kwargs):
        super(SalamanderConnectivityOptions, self).__init__()
        self.body_phase_bias = kwargs.pop("body_phase_bias", 2*np.pi/11)


class SalamanderJointsOptions(dict):
    """Salamander joints options"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, **kwargs):
        super(SalamanderJointsOptions, self).__init__()

        # Joints offsets
        self.legs_joints_offsets = [
            SalamanderOscillatorJointsOptions.legs_joints_offsets(
                joint_i,
                **kwargs
            )
            for joint_i in range(4)
        ]
        self.body_joints_offsets = (
            SalamanderOscillatorJointsOptions.body_joints_offsets(**kwargs)
        )