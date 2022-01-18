"""Simulation"""

import os
import warnings
import traceback

import numpy as np
from tqdm import tqdm

from dm_control import mjcf
from dm_control import viewer
from dm_control.rl.control import Environment, PhysicsError

import farms_pylog as pylog

from .mjcf import setup_mjcf_xml, mjcf2str
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

    def __init__(self, mjcf_model, base_link, n_iterations, timestep, **kwargs):
        super().__init__()
        self._mjcf_model = mjcf_model
        self.fast = kwargs.pop('fast', False)
        self.pause = kwargs.pop('pause', True)
        self.headless = kwargs.pop('headless', False)
        self.options = kwargs.pop('simulation_options', None)
        if self.options is not None:
            kwargs['units'] = self.options.units

        # Simulator configuration
        viewer.util._MAX_TIME_MULTIPLIER = 2**15  # pylint: disable=protected-access
        os.environ['MUJOCO_GL'] = 'egl' if self.headless else 'glfw'  # 'osmesa'
        warnings.filterwarnings('ignore', category=DeprecationWarning)

        # Simulation
        env_kwargs = extract_sub_dict(
            dictionary=kwargs,
            keys=('control_timestep', 'n_sub_steps', 'flat_observation'),
        )
        self._physics = mjcf.Physics.from_mjcf_model(mjcf_model)
        self.task = ExperimentTask(
            base_link=base_link.name,
            n_iterations=n_iterations,
            timestep=timestep,
            **kwargs,
        )
        self._env = Environment(
            physics=self._physics,
            task=self.task,
            time_limit=n_iterations*timestep,
            **env_kwargs,
        )

    @classmethod
    def from_sdf(cls, sdf_path_animat, arena_options, timestep, **kwargs):
        """From SDF"""
        mjcf_model, base_link, hfield = setup_mjcf_xml(
            sdf_path_animat=sdf_path_animat,
            arena_options=arena_options,
            timestep=timestep,
            discardvisual=kwargs.get('headless', False),
            animat_options=kwargs.get('animat_options', None),
            simulation_options=kwargs.get('simulation_options', None),
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
            hfield=hfield,
            **kwargs,
        )

    def save_mjcf_xml(self, path):
        """Save simulation to mjcf xml"""
        mjcf_xml_str = mjcf2str(mjcf_model=self._mjcf_model)
        pylog.info(mjcf_xml_str)
        with open(path, 'w+') as xml_file:
            xml_file.write(mjcf_xml_str)

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
            self.task.set_app(app=app)
            if not self.pause:
                app.toggle_pause()
            app.launch(environment_loader=self._env)
        else:
            _iterator = (
                tqdm(range(self.task.n_iterations))
                if self.options.show_progress
                else range(self.task.n_iterations)
            )
            try:
                for _ in _iterator:
                    self._env.step(action=None)
            except PhysicsError as err:
                pylog.error(traceback.format_exc())
                raise err
        pylog.info('Closing simulation')

    def postprocess(
            self,
            iteration: int,
            log_path: str = '',
            plot: bool = False,
            video: str = '',
            **kwargs,
    ):
        """Postprocessing after simulation"""

        # Times
        times = np.arange(
            0,
            self.task.timestep*self.task.n_iterations,
            self.task.timestep
        )[:iteration]

        # Log
        if log_path:
            pylog.info('Saving data to %s', log_path)
            self.task.data.to_file(
                os.path.join(log_path, 'simulation.hdf5'),
                iteration,
            )
            self.options.save(
                os.path.join(log_path, 'simulation_options.yaml')
            )
            self.task.animat_options.save(
                os.path.join(log_path, 'animat_options.yaml')
            )

        # Plot
        if plot:
            self.task.data.plot(times)

        # # Record video
        # if video and self.interface is not None:
        #     self.interface.video.save(
        #         video,
        #         iteration=iteration,
        #         writer=kwargs.pop('writer', 'ffmpeg')
        #     )
