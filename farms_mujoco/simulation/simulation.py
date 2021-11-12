"""Simulation"""

import os
import warnings

from tqdm import tqdm

from dm_control import mjcf
from dm_control import viewer
from dm_control.rl.control import Environment

import farms_pylog as pylog

from .mjcf import setup_mjcf_xml
from .task import ExperimentTask
from .application import FarmsApplication


def extract_sub_dict(dictionary, keys):
    """Extract sub-dictionary"""
    return {
        key: dictionary.pop(key)
        for key in keys
        if key in dictionary
    }


class Simulation:
    """Simulation"""

    def __init__(self, mjcf_model, base_link, duration, timestep, **kwargs):
        super().__init__()
        self._mjcf = mjcf
        self.fast = kwargs.pop('fast', False)
        self.pause = kwargs.pop('pause', True)
        self.headless = kwargs.pop('headless', False)

        # Simulator configuration
        viewer.util._MAX_TIME_MULTIPLIER = 2**15  # pylint: disable=protected-access
        os.environ['MUJOCO_GL'] = 'egl' if self.headless else 'glfw'  # 'osmesa'
        warnings.filterwarnings('ignore', category=DeprecationWarning)

        # Simulation
        env_kwargs = extract_sub_dict(
            dictionary=kwargs,
            keys=('control_timestep', 'n_sub_steps', 'flat_observation'),
        )
        self._physics = self._mjcf.Physics.from_mjcf_model(mjcf_model)
        self._task = ExperimentTask(
            base_link=base_link.name,
            duration=duration,
            timestep=timestep,
            **kwargs,
        )
        self._env = Environment(
            physics=self._physics,
            task=self._task,
            time_limit=duration,
            **env_kwargs,
        )

    @classmethod
    def from_sdf(cls, sdf_path, timestep, **kwargs):
        """From SDF"""
        mjcf_model, base_link = setup_mjcf_xml(
            sdf_path=sdf_path,
            timestep=timestep,
            **extract_sub_dict(
                dictionary=kwargs,
                keys=(
                    'spawn_position', 'spawn_rotation',
                    'save_mjcf', 'use_particles',
                ),
            )
        )
        return cls(
            mjcf_model=mjcf_model,
            base_link=base_link,
            timestep=timestep,
            **kwargs,
        )

    def physics(self):
        """Physics"""
        return self.physics

    def run(self):
        """Run simulation"""
        if not self.headless:
            app = FarmsApplication()
            app.set_speed(multiplier=(
                # pylint: disable=protected-access
                viewer.util._MAX_TIME_MULTIPLIER
                if self.fast
                else 1
            ))
            self._task.set_app(app=app)
            if not self.pause:
                app.toggle_pause()
            app.launch(environment_loader=self._env)
        else:
            for _ in tqdm(range(self._task.n_iterations)):
                self._env.step(action=None)
        pylog.info('Closing simulation')
