"""Control options"""

import numpy as np


class SalamanderControlOptions(dict):
    """Model options"""

    def __init__(self, options):
        super(SalamanderControlOptions, self).__init__()
        self.update(options)

    @classmethod
    def from_gait(cls, gait, **kwargs):
        """Salamander control option from gait"""
        return (
            cls.walking(frequency=kwargs.pop("frequency", 1), **kwargs)
            if gait == "walking"
            else cls.swimming(frequency=kwargs.pop("frequency", 2), **kwargs)
            if gait == "swimming"
            else cls.standing()
        )

    @classmethod
    def standing(cls, **kwargs):
        """Standing options"""
        # Options
        options = {}

        # General
        options["n_body_joints"] = 11
        options["frequency"] = kwargs.pop("frequency", 0)

        # Body
        options["body_amplitude_0"] = kwargs.pop("body_amplitude_0", 0)
        options["body_amplitude_1"] = kwargs.pop("body_amplitude_1", 0)
        options["body_stand_amplitude"] = kwargs.pop("body_stand_amplitude", 0)
        options["body_stand_shift"] = kwargs.pop("body_stand_shift", 0)

        # Legs
        options["leg_0_amplitude"] = kwargs.pop("leg_0_amplitude", 0)
        options["leg_0_offset"] = kwargs.pop("leg_0_offset", 0)

        options["leg_1_amplitude"] = kwargs.pop("leg_1_amplitude", 0)
        options["leg_1_offset"] = kwargs.pop("leg_1_offset", np.pi/16)

        options["leg_2_amplitude"] = kwargs.pop("leg_2_amplitude", 0)
        options["leg_2_offset"] = kwargs.pop("leg_2_offset", 0)

        options["leg_3_amplitude"] = kwargs.pop("leg_3_amplitude", 0)
        options["leg_3_offset"] = kwargs.pop("leg_3_offset", np.pi/8)

        # Additional walking options
        options["leg_turn"] = 0

        # Gains
        options["body_p"] = 1e-1
        options["body_d"] = 1e0
        options["body_f"] = 1e1
        options["legs_p"] = 1e-1
        options["legs_d"] = 1e0
        options["legs_f"] = 1e1

        # Additional options
        options.update(kwargs)
        return cls(options)

    @classmethod
    def walking(cls, **kwargs):
        """Walking options"""
        # Options
        options = {}

        # General
        options["n_body_joints"] = 11
        options["frequency"] = kwargs.pop("frequency", 1)

        # Body
        options["body_amplitude_0"] = kwargs.pop("body_amplitude_0", 0)
        options["body_amplitude_1"] = kwargs.pop("body_amplitude_1", 0)
        options["body_stand_amplitude"] = kwargs.pop(
            "body_stand_amplitude",
            0.2
        )
        options["body_stand_shift"] = kwargs.pop("body_stand_shift", np.pi/4)

        # Legs
        options["leg_0_amplitude"] = kwargs.pop("leg_0_amplitude", 0.8)
        options["leg_0_offset"] = kwargs.pop("leg_0_offset", 0)

        options["leg_1_amplitude"] = kwargs.pop("leg_1_amplitude", np.pi/32)
        options["leg_1_offset"] = kwargs.pop("leg_1_offset", np.pi/32)

        options["leg_2_amplitude"] = kwargs.pop("leg_2_amplitude", np.pi/2)
        options["leg_2_offset"] = kwargs.pop("leg_2_offset", 0)

        options["leg_3_amplitude"] = kwargs.pop("leg_3_amplitude", np.pi/8)
        options["leg_3_offset"] = kwargs.pop("leg_3_offset", np.pi/8)

        # Additional walking options
        options["leg_turn"] = 0

        # Gains
        options["body_p"] = 1e-1
        options["body_d"] = 1e0
        options["body_f"] = 1e1
        options["legs_p"] = 1e-1
        options["legs_d"] = 1e0
        options["legs_f"] = 1e1

        # Additional options
        options.update(kwargs)
        return cls(options)

    @classmethod
    def swimming(cls, **kwargs):
        """Swimming options"""
        # Options
        options = {}

        # General
        n_body_joints = 11
        options["n_body_joints"] = n_body_joints
        options["frequency"] = kwargs.pop("frequency", 2)

        # Body
        options["body_amplitude_0"] = kwargs.pop("body_amplitude_0", 0.1)
        options["body_amplitude_1"] = kwargs.pop("body_amplitude_1", 0.5)
        options["body_stand_amplitude"] = kwargs.pop("body_stand_amplitude", 0)
        options["body_stand_shift"] = kwargs.pop("body_stand_shift", 0)

        # Legs
        options["leg_0_amplitude"] = kwargs.pop("leg_0_amplitude", 0)
        options["leg_0_offset"] = kwargs.pop("leg_0_offset", -2*np.pi/5)

        options["leg_1_amplitude"] = kwargs.pop("leg_1_amplitude", 0)
        options["leg_1_offset"] = kwargs.pop("leg_1_offset", 0)

        options["leg_2_amplitude"] = kwargs.pop("leg_2_amplitude", 0)
        options["leg_2_offset"] = kwargs.pop("leg_2_offset", 0)

        options["leg_3_amplitude"] = kwargs.pop("leg_3_amplitude", 0)
        options["leg_3_offset"] = kwargs.pop("leg_3_offset", 0)

        # Additional walking options
        options["leg_turn"] = 0

        # Gains
        options["body_p"] = 1e-1
        options["body_d"] = 1e0
        options["body_f"] = 1e1
        options["legs_p"] = 1e-1
        options["legs_d"] = 1e0
        options["legs_f"] = 1e1

        # Additional options
        options.update(kwargs)
        return cls(options)

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
