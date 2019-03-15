"""Interface"""

import numpy as np

import pybullet


class DebugParameter:
    """DebugParameter"""

    def __init__(self, name, val, val_min, val_max):
        super(DebugParameter, self).__init__()
        self.name = name
        self.value = val
        self.val_min = val_min
        self.val_max = val_max
        self._handler = None
        self.add(self.value)

    def add(self, value):
        """Add parameter"""
        if self._handler is None:
            self._handler = pybullet.addUserDebugParameter(
                paramName=self.name,
                rangeMin=self.val_min,
                rangeMax=self.val_max,
                startValue=value
            )
        else:
            raise Exception(
                "Handler for parameter '{}' is already used".format(
                    self.name
                )
            )

    def remove(self):
        """Remove parameter"""
        pybullet.removeUserDebugItem(self._handler)

    def get_value(self):
        """Current value"""
        return pybullet.readUserDebugParameter(self._handler)


class ParameterPlay(DebugParameter):
    """Play/pause parameter"""

    def __init__(self):
        super(ParameterPlay, self).__init__("Play", 1, 0, 1)
        self.value = True

    def update(self):
        """Update"""
        self.value = self.get_value() > 0.5


class ParameterRTL(DebugParameter):
    """Real-time limiter"""

    def __init__(self):
        super(ParameterRTL, self).__init__("Real-time limiter", 1, 1e-3, 3)

    def update(self):
        """Update"""
        self.value = self.get_value()


class ParameterGait(DebugParameter):
    """Gait control"""

    def __init__(self, gait):
        value = 0 if gait == "standing" else 2 if gait == "swimming" else 1
        super(ParameterGait, self).__init__("Gait", value, 0, 2)
        self.value = gait
        self.changed = False

    def update(self):
        """Update"""
        previous_value = self.value
        value = self.get_value()
        self.value = (
            "standing"
            if value < 0.5
            else "walking"
            if 0.5 < value < 1.5
            else "swimming"
        )
        self.changed = (self.value != previous_value)
        if self.changed:
            print("Gait changed ({} > {})".format(
                previous_value,
                self.value
            ))


class ParameterFrequency(DebugParameter):
    """Frequency control"""

    def __init__(self, frequency):
        super(ParameterFrequency, self).__init__("Frequency", frequency, 0, 5)
        self.changed = False

    def update(self):
        """Update"""
        previous_value = self.value
        self.value = self.get_value()
        self.changed = (self.value != previous_value)
        if self.changed:
            print("frequency changed ({} > {})".format(
                previous_value,
                self.value
            ))


class ParameterBodyOffset(DebugParameter):
    """Body offset control"""

    def __init__(self):
        lim = np.pi/8
        super(ParameterBodyOffset, self).__init__("Body offset", 0, -lim, lim)
        self.changed = False

    def update(self):
        """Update"""
        previous_value = self.value
        self.value = self.get_value()
        self.changed = (self.value != previous_value)
        if self.changed:
            print("Body offset changed ({} > {})".format(
                previous_value,
                self.value
            ))


class UserParameters:
    """Parameters control"""

    def __init__(self, gait, frequency):
        super(UserParameters, self).__init__()
        self._play = ParameterPlay()
        self._rtl = ParameterRTL()
        self._gait = ParameterGait(gait)
        self._frequency = ParameterFrequency(frequency)
        self._body_offset = ParameterBodyOffset()

    def update(self):
        """Update parameters"""
        for parameter in [
                self._play,
                self._rtl,
                self._gait,
                self._frequency,
                self._body_offset
        ]:
            parameter.update()

    @property
    def play(self):
        """Play"""
        return self._play

    @property
    def rtl(self):
        """Real-time limiter"""
        return self._rtl

    @property
    def gait(self):
        """Gait"""
        return self._gait

    @property
    def frequency(self):
        """Frequency"""
        return self._frequency

    @property
    def body_offset(self):
        """Body offset"""
        return self._body_offset
