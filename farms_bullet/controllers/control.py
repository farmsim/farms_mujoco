"""Control"""

import numpy as np
import pybullet


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

    def __init__(self, model, network, joints_controllers):
        super(ModelController, self).__init__()
        self.model = model
        self.controllers = joints_controllers
        self.network = network
        self._frequency = self.controllers[0].get_frequency()
        self._body_offset = 0
        self._joint_order = [ctrl.joint() for ctrl in self.controllers]

    def control(self):
        """Control"""
        self.network.control_step()
        pybullet.setJointMotorControlArray(
            self.model,
            self._joint_order,  # [ctrl["joint"] for ctrl in controls]
            pybullet.POSITION_CONTROL,
            targetPositions=self.network.get_position_output(),  # [ctrl["cmd"]["pos"] for ctrl in controls],
            targetVelocities=self.network.get_velocity_output(),  # [ctrl["cmd"]["vel"] for ctrl in controls],
            # positionGains=[ctrl["pdf"]["p"] for ctrl in controls],
            # velocityGains=[ctrl["pdf"]["d"] for ctrl in controls],
            # forces=[ctrl["pdf"]["f"] for ctrl in controls]
        )
