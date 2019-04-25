"""Control"""

import time
import numpy as np
import pybullet

from .network import SalamanderNetworkODE
from .control_options import SalamanderControlOptions
# from .casadi import SalamanderCasADiNetwork


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

    def joint(self):
        """Joint"""
        return self._joint

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

    def __init__(self, model, joints_controllers, iterations, timestep):
        super(ModelController, self).__init__()
        self.model = model
        self.controllers = joints_controllers
        # self.network = IndependentOscillators(
        #     self.controllers,
        #     timestep=timestep
        # )
        # self.network = SalamanderNetwork.walking(timestep, phases=None)
        self.network = SalamanderNetworkODE.walking(iterations, timestep)
        self._frequency = self.controllers[0].get_frequency()
        self._body_offset = 0
        self._joint_order = [ctrl.joint() for ctrl in self.controllers]

    def control(self):
        """Control"""
        _phases = self.network.control_step()
        position = self.network.get_position_output()
        velocity = self.network.get_velocity_output()
        pybullet.setJointMotorControlArray(
            self.model,
            self._joint_order,  # [ctrl["joint"] for ctrl in controls]
            pybullet.POSITION_CONTROL,
            targetPositions=position,  # [ctrl["cmd"]["pos"] for ctrl in controls],
            targetVelocities=velocity,  # [ctrl["cmd"]["vel"] for ctrl in controls],
            # positionGains=[ctrl["pdf"]["p"] for ctrl in controls],
            # velocityGains=[ctrl["pdf"]["d"] for ctrl in controls],
            # forces=[ctrl["pdf"]["f"] for ctrl in controls]
        )


class SalamanderController(ModelController):
    """ModelController"""

    @classmethod
    def from_gait(cls, model, joints, gait, iterations, timestep, **kwargs):
        """Salamander controller from gait"""
        return cls.from_options(
            model=model,
            joints=joints,
            options=SalamanderControlOptions.from_gait(gait, **kwargs),
            iterations=iterations,
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
        self.network.update_gait(gait)

    @classmethod
    def from_options(cls, model, joints, options, iterations, timestep):
        """Salamander controller from options"""
        joint_controllers_body, joint_controllers_legs = (
            cls.joints_controllers(joints, options)
        )
        return cls(
            model,
            joint_controllers_body + joint_controllers_legs,
            iterations=iterations,
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
