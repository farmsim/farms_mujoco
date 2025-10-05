"""Simulation"""

import os
import time
import warnings
import traceback
from typing import List, Dict
from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm

import mujoco
import mujoco.viewer
from dm_control import mjcf
from dm_control import viewer as dm_viewer
from dm_control.rl.control import Environment, PhysicsError

from farms_core import pylog
from farms_core.doc import ClassDoc, ChildDoc
from farms_core.experiment.options import ExperimentOptions
from farms_core.extensions.extensions import import_item
from farms_core.sensors.data import LinkSensorArray
from farms_core.simulation.options import (
    SimulationOptions,
    ViewerExtensionOptions,
)

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


def real_time_handing(
        timestep: float,
        tic_rt: list[float],
        rtl: float = 1.0,
):
    """Real-time handling"""
    tic_rt[1] = time.time()
    tic_rt[2] += timestep/rtl - (tic_rt[1] - tic_rt[0])
    if tic_rt[2] > 2e-2:
        time.sleep(tic_rt[2])
        tic_rt[2] = 0
    elif tic_rt[2] < 0:
        tic_rt[2] = 0
    tic_rt[0] = time.time()


def create_sphere(viewer, **kwargs):
    """Create sphere"""
    scn = viewer.user_scn
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        kwargs.pop('size', [1.0, 0.0, 0.0]),  # Radius
        kwargs.pop('pos', [0.0, 0.0, 0.0]),  # Pos
        np.eye(3).ravel(),  # Matrix
        kwargs.pop('rgba', [1.0, 1.0, 1.0, 1.0]),  # RGBA
    )
    scn.ngeom += 1
    return geom


def create_line(viewer, begin, end, **kwargs):
    """Create sphere"""
    scn = viewer.user_scn
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        [1.0, 1.0, 1.0],  # Size
        begin,  # Pos
        np.eye(3).ravel(),  # Matrix
        kwargs.pop('rgba', [1.0, 0.3, 0.0, 0.7]),  # RGBA
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        kwargs.pop('width', 5),  # Width
        begin,
        end,
    )
    scn.ngeom += 1
    return geom


class MuJoCoViewerExtension(ABC):
    """MuJoCo viewer extension"""

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer

    @abstractmethod
    def iteration0(self, task):
        """Iteration 0"""

    @abstractmethod
    def step(self, iteration: int, time: float, timestep: float):
        """Step"""


class CameraFollowerViewer(MuJoCoViewerExtension):
    """Camera follower viewer"""

    def __init__(self, viewer, animat_id=0, **kwargs):
        super().__init__(viewer)
        self.links: LinkSensorArray | None = None
        self.animat_id = animat_id
        self.angular_velocity = kwargs.pop('angular_velocity', 0)  # [deg/s]

    def iteration0(self, task):
        index = self.animat_id
        self.links: LinkSensorArray = task.data.animats[index].sensors.links

    def step(self, iteration: int, time: float, timestep: float):
        del time
        if self.links is not None:
            self.viewer.cam.lookat = np.array(self.links.global_com_position(
                iteration=iteration - 1,
            ))
            self.viewer.cam.azimuth += self.angular_velocity*timestep


class CoMViewer(MuJoCoViewerExtension):
    """CoM viewer"""

    def __init__(self, viewer, **kwargs):
        super().__init__(viewer)
        self.sphere = create_sphere(
            viewer,
            size=kwargs.pop('size', [0.01, 0.0, 0.0]),
            rgba=kwargs.pop('rgba', [1.0, 1.0, 1.0, 0.3]),
        )
        self.links: LinkSensorArray | None = None

    @classmethod
    def from_options(cls, config, viewer):
        """From options"""
        return cls(
            viewer=viewer,
            size=config['size'],
            rgba=config['rgba'],
        )

    def iteration0(self, task):
        self.links: LinkSensorArray = task.data.animats[0].sensors.links
        mass = np.sum(self.links.masses)
        if mass is not None:
            radius = 0.2*((3*mass/1000)/np.pi)**(1/3)
            self.sphere.size[:] = radius

    def step(self, iteration: int, time: float, timestep: float):
        del time, timestep
        if self.links is not None:
            self.sphere.pos = np.array(self.links.global_com_position(
                iteration=iteration - 1,
            ))


class TrailCoMViewer(MuJoCoViewerExtension):
    """CoM trail viewer"""

    def __init__(self, viewer, **kwargs):
        super().__init__(viewer)
        self.pos_old = None
        self.pos_new = None
        self.width = kwargs.pop('width', 5)
        self.rgba = kwargs.pop('rgba', [1.0, 0.3, 0.0, 0.7])
        self.links: LinkSensorArray | None = None

    @classmethod
    def from_options(cls, config: ViewerExtensionOptions, viewer):
        """From options"""
        return cls(
            viewer=viewer,
            size=config['width'],
            rgba=config['rgba'],
        )

    def iteration0(self, task):
        self.links: LinkSensorArray = task.data.animats[0].sensors.links
        self.pos_new = self.pos_old = self.links.global_com_position(0)

    def step(self, iteration: int, time: float, timestep: float):
        del time, timestep
        if self.links is not None:
            self.pos_new = self.links.global_com_position(iteration-1)
            create_line(
                self.viewer,
                self.pos_old,
                self.pos_new,
                width=self.width,
                rgba=self.rgba,
            )
            self.pos_old = self.pos_new


class TrailLinkViewer(MuJoCoViewerExtension):
    """Link trail viewer"""

    def __init__(self, viewer, **kwargs):
        super().__init__(viewer)
        self.pos_old = None
        self.pos_new = None
        self.width = kwargs.pop('width', 5)
        self.rgba = kwargs.pop('rgba', [1.0, 0.3, 0.0, 0.7])
        self.link: str = kwargs.pop('link', '')
        self.link_id: int = kwargs.pop('link_id', None)
        self.links: LinkSensorArray | None = None

    @classmethod
    def from_options(cls, config: ViewerExtensionOptions, viewer):
        """From options"""
        return cls(
            viewer=viewer,
            size=config['width'],
            rgba=config['rgba'],
            link=config['link'],
        )

    def iteration0(self, task):
        self.links: LinkSensorArray = task.data.animats[0].sensors.links
        self.pos_new = self.pos_old = self.links.global_com_position(0)
        assert self.link in self.links.names, (
            f"{self.link=} not in {self.links.names=}"
        )
        self.link_id = self.links.names.index(self.link)

    def step(self, iteration: int, time: float, timestep: float):
        del time, timestep
        if self.links is not None:
            self.pos_new = self.links.com_position(iteration-1, self.link_id)
            create_line(
                self.viewer,
                self.pos_old,
                self.pos_new,
                width=5,
                rgba=self.rgba,
            )
            self.pos_old = self.pos_new


class Simulation:
    """ Simulation

    Note: Set legacy_step to False to use conventional full mj_step to update the physics with dm_control.
    It will otherwise result in incorrect computations of contact forces.
    """

    def __init__(
            self,
            mjcf_model: mjcf.element.RootElement,
            base_links: list[str],
            experiment_options: ExperimentOptions,
            legacy_step: bool = False,
            **kwargs,
    ):

        super().__init__()
        self._mjcf_model: mjcf.element.RootElement = mjcf_model
        self.options: SimulationOptions = experiment_options.simulation
        self.pause: bool = not self.options.runtime.play
        self.physics: mjcf.Physics = mjcf.Physics.from_mjcf_model(mjcf_model)
        self.handle_exceptions = kwargs.pop('handle_exceptions', False)

        # Simulator configuration
        # pylint: disable=protected-access
        dm_viewer.util._MIN_TIME_MULTIPLIER = 2**-10
        dm_viewer.util._MAX_TIME_MULTIPLIER = 2**10
        if 'MUJOCO_GL' not in os.environ:
            os.environ['MUJOCO_GL'] = (
                'egl'
                if self.options.runtime.headless
                else 'glfw'  # 'osmesa'
            )
        pylog.debug(f'Using env variable : MUJOCO_GL={os.environ["MUJOCO_GL"]}')
        warnings.filterwarnings('ignore', category=DeprecationWarning)

        # Simulation
        env_kwargs = extract_sub_dict(
            dictionary=kwargs,
            keys=('control_timestep', 'n_sub_steps', 'flat_observation'),
        )
        if 'n_sub_steps' not in env_kwargs:
            env_kwargs['n_sub_steps'] = self.options.physics.num_sub_steps
        self.task: ExperimentTask = ExperimentTask(
            experiment_options=experiment_options,
            base_links=base_links,
            n_iterations=self.options.runtime.n_iterations,
            timestep=self.options.physics.timestep,
            units=self.options.units,
            substeps=self.options.physics.cb_sub_steps,
            **kwargs,
        )

        self._env: Environment = Environment(
            physics=self.physics,
            task=self.task,
            time_limit=(
                self.options.runtime.n_iterations
                *self.options.physics.timestep
            ),
            legacy_step=legacy_step,
            **env_kwargs,
        )

        # User interface
        self.viewer_quit = False
        self.viewer_paused = not self.options.runtime.play
        self.viwer_step_iteration = False
        self.viewer_tic_rt = np.zeros(3)
        self.viewer_last_sync = 0
        self.viewer_speed = (
            2**8
            if self.options.runtime.fast
            else self.options.runtime.rtl
        )

    @property
    def iteration(self):
        """Iteration"""
        return self.task.iteration

    @classmethod
    def from_sdf(
            cls,
            experiment_options: ExperimentOptions,
            **kwargs,
    ):
        """From SDF"""
        mjcf_model, base_links, hfield = setup_mjcf_xml(
            experiment_options=experiment_options,
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
            base_links=[base_link.name for base_link in base_links],
            experiment_options=experiment_options,
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

    def update_step_options(self):
        """Update sub steps"""
        self.task.cb_sub_steps = max(1, self.task.cb_sub_steps)
        self._env._n_sub_steps = max(1, self._env._n_sub_steps)

    def viewer_callback(self, keycode):
        """UI callback"""
        code = chr(keycode)
        match code:
            case ' ':  # Space
                self.viewer_paused = not self.viewer_paused
                pylog.debug(f'Toggling pause: {self.viewer_paused=}')
            case 'Q' | 'Ā':  # ESC
                self.viewer_quit = True
                pylog.debug('Quitting viewer')
            case '=':
                self.viewer_speed *= 2
                pylog.debug(f'Simulation speed: {self.viewer_speed}')
            case '-':
                self.viewer_speed /= 2
                pylog.debug(f'Simulation speed: {self.viewer_speed}')
            case 'ĉ':  # Up
                pylog.debug('Up')
            case 'Ć':  # Right
                pylog.debug('Stepping single iteration')
                self.viwer_step_iteration = True
            case 'Ĉ':  # Down
                pylog.debug('Down')
            case 'ć':  # Left
                pylog.debug('Left')
            case _:
                pylog.debug(f'Unhandled key: "{code}" ({keycode})')

    def run(self):
        """Run simulation"""
        if not self.options.runtime.headless:
            if self.options.mujoco.viewer == 'dm_control':
                app = FarmsApplication()
                app.set_speed(multiplier=(
                    # pylint: disable=protected-access
                    dm_viewer.util._MAX_TIME_MULTIPLIER
                    if self.options.runtime.fast
                    else 1
                ))
                self.task.set_app(app=app)
                if not self.pause:
                    app.toggle_pause()
                app.launch(environment_loader=self._env)
            else:
                with mujoco.viewer.launch_passive(
                        self.physics.model.ptr,
                        self.physics.data.ptr,
                        key_callback=self.viewer_callback,
                ) as viewer:
                    iteration = 0
                    n_iterations = self.task.n_iterations
                    cb_sub_steps = self.task.cb_sub_steps
                    cam = viewer.cam
                    viewer_extentions_loaders = [
                        import_item(extension['loader'])
                        for extension in self.options.viewer_extensions
                    ]
                    viewer_extensions: list[MuJoCoViewerExtension] = [
                        loader.from_options(extension['config'], viewer)
                        for loader, extension in zip(
                                viewer_extentions_loaders,
                                self.options.viewer_extensions
                        )
                    ]
                    if not self.options.camera.free_camera:
                        viewer_extensions += [
                            CameraFollowerViewer(
                                viewer=viewer,
                                animat_id=0,
                                angular_velocity=(
                                    20
                                    if self.options.camera.rotating_camera
                                    else 0
                                ),
                            )
                        ]
                    while viewer.is_running() and iteration < n_iterations:

                        # Time
                        tic = time.time()

                        # Start simulation
                        if iteration == 0:
                            self.task.initialize_episode(self.physics, viewer)
                            viewer.opt.geomgroup[3] = 1
                            for extension in viewer_extensions:
                                extension.iteration0(self.task)

                        # Quit
                        if self.viewer_quit:
                            break

                        # Skip if paused
                        if self.viewer_paused and not self.viwer_step_iteration:
                            if tic - self.viewer_last_sync > 0.02:
                                viewer.sync()
                                self.viewer_last_sync = tic
                            continue

                        # Step
                        self.update_step_options()
                        for _ in range(cb_sub_steps):
                            self._env.step(action=None)
                        iteration += 1

                        # Pick up changes to the physics state, options from GUI
                        # FIXME Does this apply perturbations?
                        if tic - self.viewer_last_sync > 0.02:
                            viewer.sync()
                            self.viewer_last_sync = tic

                        # Viewer extensions
                        for extension in viewer_extensions:
                            extension.step(
                                iteration,
                                iteration*self.options.physics.timestep,
                                self.options.physics.timestep,
                            )

                        # Time keeping
                        real_time_handing(
                            timestep=self.options.physics.timestep,
                            tic_rt=self.viewer_tic_rt,
                            rtl=self.viewer_speed,
                        )

                        # Single simulation step
                        if self.viwer_step_iteration:
                            self.viewer_paused = True
                            self.viwer_step_iteration = False
        else:
            _iterator = (
                tqdm(range(self.task.sim_iterations))
                if self.options.runtime.show_progress
                else range(self.task.sim_iterations)
            )
            try:
                for _ in _iterator:
                    self.update_step_options()
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
            tqdm(range(self.task.n_iterations))
            if show_progress
            else range(self.task.n_iterations)
        )
        try:
            for iteration in _iterator:
                yield iteration
                self.update_step_options()
                for _ in range(self.task.cb_sub_steps):
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
            self.task.experiment_options.animats[0].save(
                os.path.join(log_path, 'animat_options.yaml')
            )

        # Plot
        if plot:
            self.task.data.plot(times)
