"""Control"""

import time

import pybullet
import numpy as np

from .network import SalamanderNetwork


class SineControl:
    """SineControl"""

    def __init__(self, amplitude, frequency, offset):
        super(SineControl, self).__init__()
        self.amplitude = amplitude
        self._angular_frequency = 2*np.pi*frequency
        self.offset = offset

    @property
    def angular_frequency(self):
        """Angular frequency"""
        return self._angular_frequency

    @angular_frequency.setter
    def angular_frequency(self, value):
        self._angular_frequency = value

    def position(self, phase):
        """"Position"""
        return self.amplitude*np.sin(phase) + self.offset

    def velocity(self, phase):
        """Velocity"""
        return self._angular_frequency*self.amplitude*np.cos(phase)


class ControlPDF(dict):
    """ControlPDF"""

    def __init__(self, p=1, d=0, f=0):
        super(ControlPDF, self).__init__()
        self["p"] = p
        self["d"] = d
        self["f"] = f

    @property
    def p_term(self):
        """Proportfonal term"""
        return self["p"]

    @property
    def d_term(self):
        """Derivative term"""
        return self["d"]

    @property
    def f_term(self):
        """Max force term"""
        return self["f"]


class JointController:
    """JointController"""

    def __init__(self, joint, sine, pdf, **kwargs):
        super(JointController, self).__init__()
        self._joint = joint
        self._sine = sine
        self._pdf = pdf
        self._is_body = kwargs.pop("is_body", False)

    def cmds(self, phase):
        """Commands"""
        return {
            "pos": self._sine.position(phase),
            "vel": self._sine.velocity(phase)
        }

    def update(self, phase):
        """Update"""
        return {
            "joint": self._joint,
            "cmd": self.cmds(phase),
            "pdf": self._pdf
        }

    def angular_frequency(self):
        """Angular frequency"""
        return self._sine.angular_frequency

    def get_frequency(self):
        """Get frequency"""
        return self._sine.angular_frequency/(2*np.pi)

    def set_frequency(self, frequency):
        """Set frequency"""
        self._sine.angular_frequency = 2*np.pi*frequency

    def set_body_offset(self, body_offset):
        """Set body offset"""
        if self._is_body:
            self._sine.offset = body_offset


class ModelController:
    """ModelController"""

    def __init__(self, model, joints_controllers, timestep):
        super(ModelController, self).__init__()
        self.model = model
        self.controllers = joints_controllers
        # self.network = IndependentOscillators(
        #     self.controllers,
        #     timestep=timestep
        # )
        self.network = SalamanderNetwork.walking(timestep, phases=None)
        self._frequency = self.controllers[0].get_frequency()
        self._body_offset = 0

    def control(self, verbose=False):
        """Control"""
        phases = self.network.control_step([
            float(controller.angular_frequency())
            for controller in self.controllers
        ])
        if verbose:
            tic = time.time()
        controls = [
            controller.update(phases[i])
            for i, controller in enumerate(self.controllers)
        ]
        if verbose:
            toc = time.time()
            print("Time to copy phases: {} [s]".format(toc-tic))
        pybullet.setJointMotorControlArray(
            self.model,
            [ctrl["joint"] for ctrl in controls],
            pybullet.POSITION_CONTROL,
            targetPositions=[ctrl["cmd"]["pos"] for ctrl in controls],
            targetVelocities=[ctrl["cmd"]["vel"] for ctrl in controls],
            positionGains=[ctrl["pdf"]["p"] for ctrl in controls],
            velocityGains=[ctrl["pdf"]["d"] for ctrl in controls],
            forces=[ctrl["pdf"]["f"] for ctrl in controls]
        )

    def update_frequency(self, frequency):
        """Update frequency"""
        self._frequency = frequency
        for controller in self.controllers:
            controller.set_frequency(frequency)

    def update_body_offset(self, body_offset):
        """Update body offset"""
        self._body_offset = body_offset
        for controller in self.controllers:
            controller.set_body_offset(body_offset)


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
        options["leg_2_offset"] = kwargs.pop("leg_2_offset", np.pi/8)

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

        options["leg_2_amplitude"] = kwargs.pop("leg_2_amplitude", np.pi/8)
        options["leg_2_offset"] = kwargs.pop("leg_2_offset", np.pi/8)

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


class SalamanderController(ModelController):
    """ModelController"""

    @classmethod
    def from_gait(cls, model, joints, gait, timestep, **kwargs):
        """Salamander controller from gait"""
        return cls.from_options(
            model=model,
            joints=joints,
            options=SalamanderControlOptions.from_gait(gait, **kwargs),
            timestep=timestep
        )

    def update_gait(self, gait, joints, timestep):
        """Update gait"""
        controllers_body, controllers_legs = (
            SalamanderController.joints_controllers(
                joints=joints,
                options=SalamanderControlOptions.from_gait(
                    gait=gait,
                    frequency=self._frequency,
                    body_offset=self._body_offset
                )
            )
        )
        self.controllers = controllers_body + controllers_legs
        self.network = SalamanderNetwork.from_gait(
            gait,
            timestep,
            phases=self.network.phases
        )

    @classmethod
    def from_options(cls, model, joints, options, timestep):
        """Salamander controller from options"""
        joint_controllers_body, joint_controllers_legs = (
            cls.joints_controllers(joints, options)
        )
        return cls(
            model,
            joint_controllers_body + joint_controllers_legs,
            timestep=timestep
        )

    @staticmethod
    def joints_controllers(joints, options):
        """Controllers"""
        n_body_joints = options["n_body_joints"]
        frequency = options["frequency"]
        amplitudes = np.linspace(
            options["body_amplitude_0"],
            options["body_amplitude_1"],
            n_body_joints
        )
        joint_controllers_body = [
            JointController(
                joint=joints["joint_link_body_{}".format(joint_i+1)],
                sine=SineControl(
                    amplitude=amplitudes[joint_i] + (
                        options["body_stand_amplitude"]*np.sin(
                            2*np.pi*joint_i/n_body_joints
                            - options["body_stand_shift"]
                        )
                    ),
                    frequency=frequency,
                    offset=0
                ),
                pdf=(
                    ControlPDF(
                        p=options["body_p"],
                        d=options["body_d"],
                        f=options["body_f"]
                    )
                ),
                is_body=True
            )
            for joint_i in range(n_body_joints)
        ]
        joint_controllers_legs = [
            JointController(
                joint=joints["joint_link_leg_{}_{}_{}".format(
                    leg_i,
                    side,
                    joint_i
                )],
                sine=SineControl(
                    amplitude=options["leg_{}_amplitude".format(joint_i)],
                    frequency=frequency,
                    # phase=(
                    #     - np.pi*np.abs(leg_i-side_i)
                    #     - options["leg_{}_phase".format(joint_i)]
                    #     + options["leg_turn"]*float(  # Turning
                    #         (0.5)*np.pi*np.sign(np.abs(leg_i-side_i) - 0.5)
                    #         if joint_i == 2
                    #         else 0
                    #     )
                    # ),
                    offset=options["leg_{}_offset".format(joint_i)]
                ),
                pdf=ControlPDF(
                    p=options["legs_p"],
                    d=options["legs_d"],
                    f=options["legs_f"]
                )
            )
            for leg_i in range(2)
            for side_i, side in enumerate(["L", "R"])
            for joint_i in range(3)
        ]
        return joint_controllers_body, joint_controllers_legs
