""" Model """

from collections import OrderedDict
from .description import DescriptionElement

import numpy as np


class Model(DescriptionElement):
    """Model"""

    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        self["name"] = kwargs.pop("name", None)
        self["pose"] = kwargs.pop("pose", "0 0 0 0 0 0")
        self["links"] = kwargs.pop("links", [])
        self["joints"] = kwargs.pop("joints", [])


class Link(DescriptionElement):
    """Link"""

    def __init__(self, **kwargs):
        super(Link, self).__init__(**kwargs)
        self["name"] = kwargs.pop("name", None)
        self["pose"] = kwargs.pop("pose", "0 0 0 0 0 0 0")
        self._pose = np.zeros(6)
        self["inertial"] = kwargs.pop("inertial", None)

    @property
    def pose(self):
        """Pose"""
        return np.copy(self._pose)

    @pose.setter
    def pose(self, value):
        self._pose = value
        sdata = " ".join(["{}" for _ in range(6)])
        self["pose"] = sdata.format(*[p for p in value])


class Joint(DescriptionElement):
    """Joint"""

    def __init__(self, **kwargs):
        super(Joint, self).__init__(**kwargs)
        self["name"] = kwargs.pop("name", None)
        self["type"] = kwargs.pop("type", None)
        self["parent"] = kwargs.pop("parent", None)
        self["child"] = kwargs.pop("child", None)
        self["axis"] = kwargs.pop("axis", None)
        self["physics"] = kwargs.pop("physics", None)
        self["frame"] = kwargs.pop("frame", None)
        self["pose"] = kwargs.pop("pose", None)
        self["sensor"] = kwargs.pop("sensor", None)


class JointCut(Joint):
    """Joint cut"""

    def __init__(self, **kwargs):
        super(JointCut, self).__init__(**kwargs)
        self["radius"] = kwargs.pop("radius", 0)

    @property
    def radius(self):
        """Joint radius"""
        return self["radius"]

    @radius.setter
    def radius(self, value):
        self["radius"] = value


def main():
    """ Main """
    model = Model(name="Model")
    model["links"].append(Link(name="link0"))
    model["joints"].append(Joint(name="joint0"))
    print(model)
    print(JointCut())


if __name__ == '__main__':
    main()
