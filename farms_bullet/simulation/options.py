"""Simulation options"""

from farms_data.options import Options
from farms_data.units import SimulationUnitScaling
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

        # Simulation
        self.timestep = kwargs.pop('timestep', 1e-3)
        self.n_iterations = kwargs.pop('n_iterations', 1000)
        self.play = kwargs.pop('play', True)
        self.fast = kwargs.pop('fast', False)
        self.headless = kwargs.pop('headless', False)
        self.show_progress = kwargs.pop('show_progress', True)

        # Camera
        self.free_camera = kwargs.pop('free_camera', False)
        self.top_camera = kwargs.pop('top_camera', False)
        self.rotating_camera = kwargs.pop('rotating_camera', False)

        # Video recording
        self.record = kwargs.pop('record', False)
        self.fps = kwargs.pop('fps', False)
        self.video_name = kwargs.pop('video_name', 'video')
        self.video_yaw = kwargs.pop('video_yaw', 0)
        self.video_pitch = kwargs.pop('video_pitch', -45)
        self.video_distance = kwargs.pop('video_distance', 1)
        self.video_filter = kwargs.pop('video_filter', None)

        # Pybullet
        self.gravity = kwargs.pop('gravity', [0, 0, -9.81])
        self.opengl2 = kwargs.pop('opengl2', False)
        self.lcp = kwargs.pop('lcp', 'dantzig')
        self.n_solver_iters = kwargs.pop('n_solver_iters', 50)
        self.erp = kwargs.pop('erp', 0)
        self.contact_erp = kwargs.pop('contact_erp', 0)
        self.friction_erp = kwargs.pop('friction_erp', 0)
        self.num_sub_steps = kwargs.pop('num_sub_steps', 0)
        self.max_num_cmd_per_1ms = kwargs.pop('max_num_cmd_per_1ms', int(1e8))
        self.residual_threshold = kwargs.pop('residual_threshold', 1e-6)
        assert not kwargs, kwargs

    def duration(self):
        """Simulation duraiton"""
        return self.n_iterations*self.timestep

    @classmethod
    def with_clargs(cls, **kwargs):
        """Create simulation options and consider command-line arguments"""
        clargs = parse_args()
        timestep = kwargs.pop('timestep', clargs.timestep)
        return cls(
            # Simulation
            timestep=timestep,
            n_iterations=kwargs.pop('n_iterations', int(clargs.duration/timestep)),
            play=kwargs.pop('play', not clargs.pause),
            fast=kwargs.pop('fast', clargs.fast),
            headless=kwargs.pop('headless', clargs.headless),
            show_progress=kwargs.pop('show_progress', clargs.show_progress),

            # Units
            meters=kwargs.pop('meters', clargs.meters),
            seconds=kwargs.pop('seconds', clargs.seconds),
            kilograms=kwargs.pop('kilograms', clargs.kilograms),

            # Camera
            free_camera=kwargs.pop('free_camera', clargs.free_camera),
            top_camera=kwargs.pop('top_camera', clargs.top_camera),
            rotating_camera=kwargs.pop('rotating_camera', clargs.rotating_camera),

            # Video recording
            record=kwargs.pop('record', clargs.record),
            fps=kwargs.pop('fps', clargs.fps),
            video_yaw=kwargs.pop('video_yaw', clargs.video_yaw),
            video_pitch=kwargs.pop('video_pitch', clargs.video_pitch),
            video_distance=kwargs.pop('video_distance', clargs.video_distance),
            video_filter=kwargs.pop('video_filter', clargs.video_motion_filter),

            # Pybullet
            gravity=kwargs.pop('gravity', clargs.gravity),
            opengl2=kwargs.pop('opengl2', clargs.opengl2),
            lcp=kwargs.pop('lcp', clargs.lcp),
            n_solver_iters=kwargs.pop('n_solver_iters', clargs.n_solver_iters),
            erp=kwargs.pop('erp', clargs.erp),
            contact_erp=kwargs.pop('contact_erp', clargs.contact_erp),
            friction_erp=kwargs.pop('friction_erp', clargs.friction_erp),
            num_sub_steps=kwargs.pop('num_sub_steps', clargs.num_sub_steps),
            max_num_cmd_per_1ms=kwargs.pop(
                'max_num_cmd_per_1ms',
                clargs.max_num_cmd_per_1ms
            ),
            residual_threshold=kwargs.pop(
                'residual_threshold',
                clargs.residual_threshold
            ),
            **kwargs,
        )
