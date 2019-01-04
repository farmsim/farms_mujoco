""" Generate controller """

from collections import OrderedDict
import numpy as np


def control_data(data):
    """ Data """
    assert isinstance(data, OrderedDict)
    return OrderedDict([
        (
            name,
            (
                control_data(data[name])
                if isinstance(data[name], OrderedDict)
                else data[name]
            )
        )
        for name in data
    ])


class ControlPID(OrderedDict):
    """ ControlPID """

    def __init__(self, p, i, d):
        super(ControlPID, self).__init__()
        self["p"] = p
        self["i"] = i
        self["d"] = d

    @property
    def p_term(self):
        """ Proportional term """
        return self["p"]

    @p_term.setter
    def p_term(self, value):
        assert value >= 0
        self["p"] = value

    @property
    def i_term(self):
        """ Integrator term """
        return self["i"]

    @i_term.setter
    def i_term(self, value):
        assert value >= 0
        self["i"] = value

    @property
    def d_term(self):
        """ Derivative term """
        return self["d"]

    @d_term.setter
    def d_term(self, value):
        assert value >= 0
        self["d"] = value


class ControlPIDs(OrderedDict):
    """ ControlPIDs """

    def __init__(self, position, velocity):
        super(ControlPIDs, self).__init__()
        self["position"] = position
        self["velocity"] = velocity

    @property
    def position(self):
        """ Position """
        return self["position"]

    @position.setter
    def position(self, value):
        self["position"] = value

    @property
    def velocity(self):
        """ Velocity """
        return self["velocity"]

    @velocity.setter
    def velocity(self, value):
        self["velocity"] = value


class ControlOscillator(OrderedDict):
    """ ControlOscillator """

    def __init__(self, amplitude, frequency, phase, bias):
        super(ControlOscillator, self).__init__()
        self["amplitude"] = amplitude
        self["frequency"] = frequency
        self["phase"] = phase
        self["bias"] = bias

    @property
    def amplitude(self):
        """ Amplitude """
        return self["amplitude"]

    @amplitude.setter
    def amplitude(self, value):
        self["amplitude"] = value

    @property
    def frequency(self):
        """ Frequency """
        return self["frequency"]

    @frequency.setter
    def frequency(self, value):
        self["frequency"] = value

    @property
    def phase(self):
        """ Phase """
        return self["phase"]

    @phase.setter
    def phase(self, value):
        self["phase"] = value

    @property
    def bias(self):
        """ Bias """
        return self["bias"]

    @bias.setter
    def bias(self, value):
        self["bias"] = value


class ControlJoint(OrderedDict):
    """ ControlJoint """

    def __init__(self, **kwargs):
        super(ControlJoint, self).__init__()
        self["type"] = kwargs.pop("type", "position")
        self["oscillator"] = kwargs.pop("oscillator", ControlOscillator(
            amplitude=0,
            frequency=0,
            phase=0,
            bias=0
        ))
        self["pid"] = kwargs.pop("pid", ControlPIDs(
            position=kwargs.pop("position", ControlPID(p=0, i=0, d=0)),
            velocity=kwargs.pop("velocity", ControlPID(p=0, i=0, d=0))
        ))

    @property
    def type(self):
        """ Type """
        return self["type"]

    @type.setter
    def type(self, value):
        self["type"] = value

    @property
    def oscillator(self):
        """ Oscillator """
        return self["oscillator"]

    @oscillator.setter
    def oscillator(self, value):
        self["oscillator"] = value

    @property
    def pid(self):
        """ Pid """
        return self["pid"]

    @pid.setter
    def pid(self, value):
        self["pid"] = value


class ControlJoints(OrderedDict):
    """ ControlJoints """

    def __init__(self, gait, frequency, body_bias, **kwargs):
        super(ControlJoints, self).__init__()
        # Body
        n_body = kwargs.pop("n_body", 11)
        n_legs = kwargs.pop("n_legs", 2)
        for i in range(n_body):
            amplitude = 0.3 if gait == "walking" else 0.1+i*0.4/n_body
            self["link_body_{}".format(i+1)] = ControlJoint(
                type="position",
                oscillator=ControlOscillator(
                    amplitude=(
                        float(amplitude*np.sin(2*np.pi*i/n_body))
                        if gait == "walking"
                        else float(amplitude)
                    ),
                    frequency=frequency,
                    phase=0 if gait == "walking" else float(2*np.pi*i/n_body),
                    bias=body_bias
                ),
                pid=ControlPIDs(
                    position=ControlPID(
                        p=1e1 if gait == "walking" else 1e1,
                        i=1e0 if gait == "walking" else 0,
                        d=0
                    ),
                    velocity=ControlPID(
                        p=1e-2 if gait == "walking" else 1e-2,
                        i=1e-3 if gait == "walking" else 0,
                        d=0
                    )
                )
            )
        # Legs
        for leg_i in range(n_legs):
            for side_i, side in enumerate(["L", "R"]):
                for part_i in range(3):
                    name = "link_leg_{}_{}_{}".format(
                        leg_i,
                        side,
                        part_i
                    )
                    self[name] = ControlJoint(
                        type="position",
                        oscillator=ControlOscillator(
                            amplitude=(
                                float(0.6 if part_i == 0 else 0.1)
                                if gait == "walking"
                                else 0.0
                            ),
                            frequency=(
                                float(frequency)
                                if gait == "walking"
                                else 0
                            ),
                            phase=(
                                float(
                                    np.pi*np.abs(leg_i-side_i)
                                    + (0 if part_i == 0 else 0.5*np.pi)
                                )
                                if gait == "walking"
                                else 0
                            ),
                            bias=(
                                float(0 if part_i == 0 else 0.1)
                                if gait == "walking"
                                else -2*np.pi/5 if part_i == 0
                                else 0
                            )
                        ),
                        pid=ControlPIDs(
                            position=ControlPID(
                                p=1e1,
                                i=1e0,
                                d=0
                            ),
                            velocity=ControlPID(
                                p=1e-3,
                                i=1e-4,
                                d=0
                            )
                        )
                    )

    @property
    def joints(self):
        """ Joints """
        return self["joints"]

    @joints.setter
    def joints(self, value):
        self["joints"] = value


class ControlParameters(OrderedDict):
    """ Control parameters """

    def __init__(self, gait, frequency, **kwargs):
        super(ControlParameters, self).__init__()
        self.gait = gait
        self.frequency = frequency
        self.body_bias = kwargs.pop("body_bias", 0)
        self.n_body = kwargs.pop("n_body", 11)
        self.n_legs = kwargs.pop("n_legs", 2)
        self["joints"] = kwargs.pop(
            "joints",
            ControlJoints(
                self.gait,
                self.frequency,
                self.body_bias,
                n_body=self.n_body,
                n_legs=self.n_legs
            )
        )

    def data(self):
        """ Data """
        return control_data(self)
