"""Simulation options"""

import yaml
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

    def to_dict(self):
        """To dictionary"""
        return {
            key: value.to_dict() if isinstance(value, Options) else value
            for key, value in self.items()
        }

    @classmethod
    def load(cls, filename):
        """Load from file"""
        with open(filename, 'r') as yaml_file:
            options = yaml.full_load(yaml_file)
        return cls(**options)

    def save(self, filename):
        """Save to file"""
        with open(filename, 'w+') as yaml_file:
            yaml.dump(
                self.to_dict(),
                yaml_file,
                default_flow_style=False
            )


class SimulationUnitScaling(Options):
    """Simulation scaling

    1 [m] in reality = self.meterss [m] in simulation
    1 [s] in reality = self.seconds [s] in simulation
    1 [kg] in reality = self.kilograms [kg] in simulation

    """

    def __init__(self, meters=1, seconds=1, kilograms=1):
        super(SimulationUnitScaling, self).__init__()
        self.meters = meters
        self.seconds = seconds
        self.kilograms = kilograms

    @property
    def hertz(self):
        """Hertz (frequency)

        Scaled as self.hertz = 1/self.seconds

        """
        return 1./self.seconds

    @property
    def newtons(self):
        """Newtons

        Scaled as self.newtons = self.kilograms*self.meters/self.time**2

        """
        return self.kilograms*self.acceleration

    @property
    def torques(self):
        """Torques

        Scaled as self.torques = self.kilograms*self.meters**2/self.time**2

        """
        return self.newtons*self.meters

    @property
    def velocity(self):
        """Velocity

        Scaled as self.velocities = self.meters/self.seconds

        """
        return self.meters/self.seconds

    @property
    def acceleration(self):
        """Acceleration

        Scaled as self.gravity = self.meters/self.seconds**2

        """
        return self.velocity/self.seconds

    @property
    def gravity(self):
        """Gravity

        Scaled as self.gravity = self.meters/self.seconds**2

        """
        return self.acceleration

    @property
    def volume(self):
        """Volume

        Scaled as self.volume = self.meters**3

        """
        return self.meters**3

    @property
    def density(self):
        """Density

        Scaled as self.density = self.kilograms/self.meters**3

        """
        return self.kilograms/self.volume


class SimulationOptions(Options):
    """Simulation options"""

    def __init__(self, **kwargs):
        super(SimulationOptions, self).__init__()
        units = kwargs.pop('units', None)
        self.units = SimulationUnitScaling(
            meters=units.pop('meters', 1),
            seconds=units.pop('seconds', 1),
            kilograms=units.pop('kilograms', 1)
        ) if isinstance(units, dict) else SimulationUnitScaling(
            meters=kwargs.pop('meters', 1),
            seconds=kwargs.pop('seconds', 1),
            kilograms=kwargs.pop('kilograms', 1)
        )
        self.timestep = kwargs.pop('timestep', 1e-3)
        self.n_iterations = kwargs.pop('n_iterations', 100)
        self.n_solver_iters = kwargs.pop('n_solver_iters', 50)
        self.free_camera = kwargs.pop('free_camera', False)
        self.rotating_camera = kwargs.pop('rotating_camera', False)
        self.top_camera = kwargs.pop('top_camera', False)
        self.fast = kwargs.pop('fast', False)
        self.record = kwargs.pop('record', False)
        self.fps = kwargs.pop('fps', False)
        self.video_name = kwargs.pop('video_name', 'video')
        self.video_yaw = kwargs.pop('video_yaw', 0)
        self.video_pitch = kwargs.pop('video_pitch', -45)
        self.video_distance = kwargs.pop('video_distance', 1)
        self.headless = kwargs.pop('headless', False)
        self.gravity = kwargs.pop('gravity', [0, 0, -9.81])
        assert not kwargs, kwargs

    def duration(self):
        """Simulation duraiton"""
        return self.n_iterations * self.timestep

    @classmethod
    def with_clargs(cls, **kwargs):
        """Create simulation options and consider command-line arguments"""
        clargs = parse_args()
        timestep = kwargs.pop('timestep', clargs.timestep)
        return cls(
            timestep=timestep,
            n_iterations=kwargs.pop('n_iterations', int(clargs.duration/timestep)),
            n_solver_iters=kwargs.pop('n_solver_iters', clargs.n_solver_iters),
            free_camera=kwargs.pop('free_camera', clargs.free_camera),
            rotating_camera=kwargs.pop('rotating_camera', clargs.rotating_camera),
            top_camera=kwargs.pop('top_camera', clargs.top_camera),
            fast=kwargs.pop('fast', clargs.fast),
            record=kwargs.pop('record', clargs.record),
            fps=kwargs.pop('fps', clargs.fps),
            video_yaw=kwargs.pop('video_yaw', clargs.video_yaw),
            video_pitch=kwargs.pop('video_pitch', clargs.video_pitch),
            video_distance=kwargs.pop('video_distance', clargs.video_distance),
            headless=kwargs.pop('headless', clargs.headless),
            **kwargs,
        )
