"""Simulation options"""

from .parse_args import parse_args


class SimulationOptions(dict):
    """Simulation options"""

    def __init__(self, **kwargs):
        super(SimulationOptions, self).__init__()
        self["timestep"] = kwargs.pop("timestep", 1e-3)
        self["duration"] = kwargs.pop("duration", 10)
        self["gait"] = kwargs.pop("gait", "walking")
        self["free_camera"] = kwargs.pop("free_camera", False)
        self["rotating_camera"] = kwargs.pop("rotating_camera", False)
        self["top_camera"] = kwargs.pop("top_camera", False)
        self["fast"] = kwargs.pop("fast", False)
        self["record"] = kwargs.pop("record", False)

    @classmethod
    def with_clargs(cls, **kwargs):
        """Create simulation options and consider command-line arguments"""
        clargs = parse_args()
        return cls(
            free_camera=kwargs.pop("free_camera", clargs.free_camera),
            rotating_camera=kwargs.pop(
                "rotating_camera",
                clargs.rotating_camera
            ),
            top_camera=kwargs.pop("top_camera", clargs.top_camera),
            fast=kwargs.pop("fast", clargs.fast),
            record=kwargs.pop("record", clargs.record),
            **kwargs
        )

    @property
    def timestep(self):
        """Simulation timestep"""
        return self["timestep"]

    @property
    def duration(self):
        """Simulation duration"""
        return self["duration"]

    @property
    def gait(self):
        """Model gait"""
        return self["gait"]

    @property
    def free_camera(self):
        """Use a free camera during simulation"""
        return self["free_camera"]

    @property
    def rotating_camera(self):
        """Use a rotating camera during simulation"""
        return self["rotating_camera"]

    @property
    def top_camera(self):
        """Use a top view camera during simulation"""
        return self["top_camera"]

    @property
    def fast(self):
        """Disable real-time simulation and run as fast as possible"""
        return self["fast"]

    @property
    def record(self):
        """Record simulation to video"""
        return self["record"]
