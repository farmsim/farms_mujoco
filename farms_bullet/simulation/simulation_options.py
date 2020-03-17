"""Simulation options"""

from farms_sdf.units import SimulationUnitScaling
from .parse_args import parse_args


class Options(dict):
    """Options"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __getstate__(self):
        """Get state"""
        return self

    def __setstate__(self, value):
        """Get state"""
        for item in value:
            self[item] = value[item]


class SimulationOptions(Options):
    """Simulation options"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, **kwargs):
        super(SimulationOptions, self).__init__()
        self.units = SimulationUnitScaling(
            meters=kwargs.pop("meters_scaling", 1),
            seconds=kwargs.pop("seconds_scaling", 1),
            kilograms=kwargs.pop("kilograms_scaling", 1)
        )
        self.timestep = kwargs.pop("timestep", 1e-3)
        self.duration = kwargs.pop("duration", 100)
        self.n_solver_iters = kwargs.pop("n_solver_iters", 50)
        self.free_camera = kwargs.pop("free_camera", False)
        self.rotating_camera = kwargs.pop("rotating_camera", False)
        self.top_camera = kwargs.pop("top_camera", False)
        self.fast = kwargs.pop("fast", False)
        self.record = kwargs.pop("record", False)
        self.video_name = kwargs.pop("video_name", "video")
        self.video_yaw = kwargs.pop("video_yaw", 0)
        self.video_pitch = kwargs.pop("video_pitch", -45)
        self.video_distance = kwargs.pop("video_distance", 1)
        self.headless = kwargs.pop("headless", False)
        self.frequency = kwargs.pop("frequency", 1)
        self.body_stand_amplitude = kwargs.pop("body_stand_amplitude", 0.2)
        self.plot = kwargs.pop("plot", True)
        self.log_path = kwargs.pop("log_path", False)
        self.log_extension = kwargs.pop("log_extension", "npy")
        self.arena = kwargs.pop("arena", "floor")  # "water"

    @property
    def n_iterations(self):
        """Number of simulation iterations"""
        return int(self.duration / self.timestep)

    @classmethod
    def with_clargs(cls, **kwargs):
        """Create simulation options and consider command-line arguments"""
        clargs = parse_args()
        return cls(
            timestep=kwargs.pop("timestep", clargs.timestep),
            duration=kwargs.pop("duration", clargs.duration),
            n_solver_iters=kwargs.pop("n_solver_iters", clargs.n_solver_iters),
            free_camera=kwargs.pop("free_camera", clargs.free_camera),
            rotating_camera=kwargs.pop(
                "rotating_camera",
                clargs.rotating_camera
            ),
            top_camera=kwargs.pop("top_camera", clargs.top_camera),
            fast=kwargs.pop("fast", clargs.fast),
            record=kwargs.pop("record", clargs.record),
            headless=kwargs.pop("headless", clargs.headless),
            plot=kwargs.pop("plot", clargs.plot),
            log_path=kwargs.pop("log_path", clargs.log_path),
            log_extension=kwargs.pop("log_extension", clargs.log_extension),
            **kwargs
        )

    @property
    def frequency(self):
        """Model frequency"""
        return self["frequency"]

    @property
    def body_stand_amplitude(self):
        """Model body amplitude"""
        return self["body_stand_amplitude"]

    @property
    def timestep(self):
        """Simulation timestep"""
        return self["timestep"]

    @property
    def duration(self):
        """Simulation duration"""
        return self["duration"]

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

    @property
    def headless(self):
        """Headless simulation instead of using GUI"""
        return self["headless"]

    @property
    def plot(self):
        """Plot at end of experiment for results analysis"""
        return self["plot"]

    @property
    def log_path(self):
        """Log at end of experiment for results analysis"""
        return self["log_path"]

    @property
    def log_extension(self):
        """Logs files extention"""
        return self["log_extension"]
