"""Simulation options"""

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
