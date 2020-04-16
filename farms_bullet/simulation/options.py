"""Simulation options"""

import yaml
from farms_data.options import Options
from farms_sdf.units import SimulationUnitScaling
from .parse_args import parse_args


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
        self.video_name = kwargs.pop('video_name', 'video')
        self.video_yaw = kwargs.pop('video_yaw', 0)
        self.video_pitch = kwargs.pop('video_pitch', -45)
        self.video_distance = kwargs.pop('video_distance', 1)
        self.headless = kwargs.pop('headless', False)
        self.frequency = kwargs.pop('frequency', 1)
        self.plot = kwargs.pop('plot', True)
        self.log_path = kwargs.pop('log_path', False)
        self.log_extension = kwargs.pop('log_extension', 'npy')
        self.arena = kwargs.pop('arena', 'floor')  # 'water'
        self.gravity = kwargs.pop('gravity', [0, 0, -9.81])

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
            n_iterations=kwargs.pop('duration', int(clargs.duration/timestep)),
            n_solver_iters=kwargs.pop('n_solver_iters', clargs.n_solver_iters),
            free_camera=kwargs.pop('free_camera', clargs.free_camera),
            rotating_camera=kwargs.pop(
                'rotating_camera',
                clargs.rotating_camera
            ),
            top_camera=kwargs.pop('top_camera', clargs.top_camera),
            fast=kwargs.pop('fast', clargs.fast),
            record=kwargs.pop('record', clargs.record),
            headless=kwargs.pop('headless', clargs.headless),
            plot=kwargs.pop('plot', clargs.plot),
            log_path=kwargs.pop('log_path', clargs.log_path),
            log_extension=kwargs.pop('log_extension', clargs.log_extension),
            **kwargs
        )

    # def frequency(self):
    #     """Model frequency"""
    #     return self['frequency']

    # def timestep(self):
    #     """Simulation timestep"""
    #     return self['timestep']

    # def duration(self):
    #     """Simulation duration"""
    #     return self['duration']

    # def free_camera(self):
    #     """Use a free camera during simulation"""
    #     return self['free_camera']

    # def rotating_camera(self):
    #     """Use a rotating camera during simulation"""
    #     return self['rotating_camera']

    # def top_camera(self):
    #     """Use a top view camera during simulation"""
    #     return self['top_camera']

    # def fast(self):
    #     """Disable real-time simulation and run as fast as possible"""
    #     return self['fast']

    # def record(self):
    #     """Record simulation to video"""
    #     return self['record']

    # def headless(self):
    #     """Headless simulation instead of using GUI"""
    #     return self['headless']

    # def plot(self):
    #     """Plot at end of experiment for results analysis"""
    #     return self['plot']

    # def log_path(self):
    #     """Log at end of experiment for results analysis"""
    #     return self['log_path']

    # def log_extension(self):
    #     """Logs files extention"""
    #     return self['log_extension']
