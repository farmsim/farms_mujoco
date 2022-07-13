"""Simulation"""

import os
import warnings
import traceback
from typing import List, Dict

import numpy as np
from tqdm import tqdm

from dm_control import mjcf
from dm_control import viewer
from dm_control.rl.control import Environment, PhysicsError

from farms_core import pylog
from farms_core.model.options import AnimatOptions, ArenaOptions
from farms_core.simulation.options import SimulationOptions

from .mjcf import setup_mjcf_xml, mjcf2str
from .task import ExperimentTask
from .application import FarmsApplication


def extract_sub_dict(dictionary: Dict, keys: List[str]) -> Dict:
    """Extract sub-dictionary"""
    return {
        key: dictionary.pop(key)
        for key in keys
        if key in dictionary
    }


class Simulation:
    """Simulation"""

    def __init__(
            self,
            mjcf_model: mjcf.element.RootElement,
            base_link: str,
            simulation_options: SimulationOptions,
            **kwargs,
    ):
        super().__init__()
        self._mjcf_model: mjcf.element.RootElement = mjcf_model
        self.options: SimulationOptions = simulation_options
        self.pause: bool = not self.options.play
        self._physics: mjcf.Physics = mjcf.Physics.from_mjcf_model(mjcf_model)
        self.handle_exceptions = kwargs.pop('handle_exceptions', False)

        # Simulator configuration
        # pylint: disable=protected-access
        viewer.util._MIN_TIME_MULTIPLIER = 2**-10
        viewer.util._MAX_TIME_MULTIPLIER = 2**10
        os.environ['MUJOCO_GL'] = (
            'egl'
            if self.options.headless
            else 'glfw'  # 'osmesa'
        )
        warnings.filterwarnings('ignore', category=DeprecationWarning)

        # Simulation
        env_kwargs = extract_sub_dict(
            dictionary=kwargs,
            keys=('control_timestep', 'n_sub_steps', 'flat_observation'),
        )
        self.task: ExperimentTask = ExperimentTask(
            base_link=base_link,
            n_iterations=self.options.n_iterations,
            timestep=self.options.timestep,
            units=self.options.units,
            substeps=self.options.num_sub_steps,
            **kwargs,
        )
        self._env: Environment = Environment(
            physics=self._physics,
            task=self.task,
            time_limit=self.options.n_iterations*self.options.timestep,
            **env_kwargs,
        )

    @property
    def iteration(self):
        """Iteration"""
        return self.task.iteration

    @classmethod
    def from_sdf(
            cls,
            simulation_options: SimulationOptions,
            animat_options: AnimatOptions,
            arena_options: ArenaOptions,
            **kwargs,
    ):
        """From SDF"""
        substeps = max(1, simulation_options.num_sub_steps)
        mjcf_model, base_link, hfield = setup_mjcf_xml(
            timestep=simulation_options.timestep/substeps,
            discardvisual=simulation_options.headless,
            simulation_options=simulation_options,
            animat_options=animat_options,
            arena_options=arena_options,
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
            base_link=base_link.name,
            simulation_options=simulation_options,
            animat_options=animat_options,
            hfield=hfield,
            **kwargs,
        )

    def save_mjcf_xml(self, path: str, verbose: bool = False):
        """Save simulation to mjcf xml"""
        mjcf_xml_str = mjcf2str(mjcf_model=self._mjcf_model)
        if verbose:
            pylog.info(mjcf_xml_str)
        with open(path, 'w+', encoding='utf-8') as xml_file:
            xml_file.write(mjcf_xml_str)

    def run(self):
        """Run simulation"""
        if not self.options.headless:
            app = FarmsApplication()
            app.set_speed(multiplier=(
                # pylint: disable=protected-access
                viewer.util._MAX_TIME_MULTIPLIER
                if self.options.fast
                else 1
            ))
            self.task.set_app(app=app)
            if not self.pause:
                app.toggle_pause()
            app.launch(environment_loader=self._env)
        else:
            _iterator = (
                tqdm(range(self.task.sim_iterations))
                if self.options.show_progress
                else range(self.task.sim_iterations)
            )
            try:
                for _ in _iterator:
                    self._env.step(action=None)
            except PhysicsError as err:
                pylog.error(traceback.format_exc())
                if self.handle_exceptions:
                    return
                raise err
        pylog.info('Closing simulation')

    def iterator(self, show_progress: bool = True, verbose: bool = True):
        """Run simulation"""
        _iterator = (
            tqdm(range(self.task.n_iterations+1))
            if show_progress
            else range(self.task.n_iterations+1)
        )
        try:
            for iteration in _iterator:
                yield iteration
                self._env.step(action=None)
        except PhysicsError as err:
            if verbose:
                pylog.error(traceback.format_exc())
            raise err

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
            self.task.timestep,
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
