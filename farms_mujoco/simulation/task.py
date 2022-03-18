"""Task"""

from typing import List, Dict

import numpy as np

from dm_control.rl.control import Task
from dm_control.mujoco.wrapper import mjbindings
from dm_control.viewer.application import Application
from dm_control.mjcf.physics import Physics

import farms_pylog as pylog
from farms_data.model.options import ModelOptions
from farms_data.model.control import ControlType, ModelController
from farms_data.amphibious.animat_data import ModelData
from farms_data.units import SimulationUnitScaling as SimulationUnits

from .physics import (
    get_sensor_maps,
    get_physics2data_maps,
    physics2data,
)


def duration2nit(duration: float, timestep: float) -> int:
    """Number of iterations from duration"""
    return int(duration/timestep)


class ExperimentTask(Task):
    """FARMS experiment"""

    def __init__(
            self,
            base_link: str,
            n_iterations: int,
            timestep: float,
            **kwargs,
    ):
        super().__init__()
        self._app: Application = None
        self.iteration: int = 0
        self.timestep: float = timestep
        self.n_iterations: int = n_iterations
        self.base_link: str = base_link
        self.data: ModelData = kwargs.pop('data', None)
        self._controller: ModelController = kwargs.pop('controller', None)
        self.animat_options: ModelOptions = kwargs.pop('animat_options', None)
        self.maps: Dict = {
            'sensors': {}, 'ctrl': {},
            'xpos': {}, 'qpos': {}, 'xfrc': {}, 'geoms': {},
            'links': {}, 'joints': {}, 'contacts': {}, 'hydrodynamics': {},
        }
        self.external_force: float = kwargs.pop('external_force', 0.2)
        self._restart: bool = kwargs.pop('restart', True)
        self._callbacks: List[TaskCallback] = kwargs.pop('callbacks', [])
        self._extras: Dict = {'hfield': kwargs.pop('hfield', None)}
        self.units: SimulationUnits = kwargs.pop('units', SimulationUnits())
        assert not kwargs, kwargs

    def set_app(self, app: Application):
        """Set application"""
        assert isinstance(app, Application)
        self._app = app

    def initialize_episode(self, physics: Physics):
        """Sets the state of the environment at the start of each episode"""

        # Checks
        if self._restart:
            assert self._app is not None, (
                'Simulation can not be restarted without application interface'
            )

        # Initialise iterations
        self.iteration = 0

        # Initialise terrain
        if self._extras['hfield'] is not None:
            data = self._extras['hfield']['data']
            hfield = self._extras['hfield']['asset']
            nrow = physics.bind(hfield).nrow
            ncol = physics.bind(hfield).ncol
            idx0 = physics.bind(hfield).adr
            size = nrow*ncol
            physics.model.hfield_data[idx0:idx0+size] = 2*(data.flatten()-0.5)
            if physics.contexts:
                with physics.contexts.gl.make_current() as ctx:
                    ctx.call(
                        mjbindings.mjlib.mjr_uploadHField,
                        physics.model.ptr,
                        physics.contexts.mujoco.ptr,
                        physics.bind(hfield).element_id,
                    )

        # Maps, data and sensors
        self.initialize_maps(physics)
        if self.data is None:
            pylog.info('No data provided, initialising default')
            self.initialize_data()
        self.initialize_sensors(physics)

        # Control
        if self._controller is not None:
            self.initialize_control(physics)

        # Initialize joints
        for joint in self.animat_options.morphology.joints:
            assert joint.name in self.maps['qpos']['names']
            index = self.maps['qpos']['names'].index(joint.name)+6
            physics.data.qpos[index] = joint.initial_position
            physics.data.qvel[index-1] = joint.initial_velocity

        if self._app is not None:
            cam = self._app._viewer.camera  # pylint: disable=protected-access
            links = self.data.sensors.links
            cam.look_at(
                position=links.urdf_position(iteration=0, link_i=0),
                distance=3,
            )

        # Callbacks
        for callback in self._callbacks:
            callback.initialize_episode(task=self, physics=physics)

    def before_step(self, action, physics: Physics):
        """Operations before physics step"""

        # Checks
        assert self.iteration < self.n_iterations

        # Sensors
        physics2data(
            physics=physics,
            iteration=self.iteration,
            data=self.data,
            maps=self.maps,
            units=self.units,
        )

        # Callbacks
        for callback in self._callbacks:
            callback.before_step(task=self, action=action, physics=physics)

        # Control
        if self._controller is not None:
            self.step_control(physics)

    def initialize_maps(self, physics: Physics):
        """Initialise data"""
        physics_named = physics.named.data
        # Links indices
        self.maps['xpos']['names'] = physics_named.xpos.axes.row.names
        # Joints indices
        self.maps['qpos']['names'] = physics_named.qpos.axes.row.names
        # External forces indices
        self.maps['xfrc']['names'] = physics_named.xfrc_applied.axes.row.names
        # Geoms indices
        self.maps['geoms']['names'] = physics_named.geom_xpos.axes.row.names

    def initialize_data(self):
        """Initialise data"""
        self.data = ModelData.from_sensors_names(
            timestep=self.timestep,
            n_iterations=self.n_iterations,
            links=self.maps['xpos']['names'],
            joints=self.maps['qpos']['names'],
            # contacts=[],
            # hydrodynamics=[],
        )

    def initialize_sensors(self, physics: Physics):
        """Initialise sensors"""
        self.maps['sensors'] = get_sensor_maps(physics)
        get_physics2data_maps(
            physics=physics,
            sensor_data=self.data.sensors,
            sensor_maps=self.maps['sensors'],
        )

    def initialize_control(self, physics: Physics):
        """Initialise controller"""
        ctrl_names = np.array(physics.named.data.ctrl.axes.row.names)
        for joint in self._controller.joints_names[ControlType.POSITION]:
            assert f'actuator_position_{joint}' in ctrl_names, (
                f'{joint} not in {ctrl_names}'
            )

        # Joints maps
        self.maps['ctrl']['pos'] = [
            np.argwhere(ctrl_names == f'actuator_position_{joint}')[0, 0]
            for joint in self._controller.joints_names[ControlType.POSITION]
        ]
        self.maps['ctrl']['vel'] = [
            np.argwhere(ctrl_names == f'actuator_velocity_{joint}')[0, 0]
            for joint in self._controller.joints_names[ControlType.VELOCITY]
        ]
        self.maps['ctrl']['trq'] = [
            np.argwhere(ctrl_names == f'actuator_torque_{joint}')[0, 0]
            for joint in self._controller.joints_names[ControlType.TORQUE]
        ]
        act_trnid = physics.named.model.actuator_trnid
        jnt_names = physics.named.model.jnt_type.axes.row.names
        jntname2actid = {name: {} for name in jnt_names}
        for act_i, act_bias in enumerate(physics.model.actuator_biasprm):
            act_type = (
                'pos' if act_bias[1] != 0
                else 'vel' if act_bias[2] != 0
                else 'trq'
            )
            jnt_name = jnt_names[act_trnid[act_i][0]]
            jntname2actid[jnt_name][act_type] = act_i

        # Actuator limits
        if self.animat_options is not None:
            animat_options = self.animat_options
            for jnt_opts in animat_options.control.joints:
                jnt_name = jnt_opts['joint_name']
                if 'position' not in jnt_opts.control_types:
                    for act_type in ('pos', 'vel'):
                        if act_type in jntname2actid[jnt_name]:
                            physics.named.model.actuator_forcelimited[
                                jntname2actid[jnt_name][act_type]
                            ] = True
                            physics.named.model.actuator_forcerange[
                                jntname2actid[jnt_name][act_type]
                            ] = [0, 0]

    def step_control(self, physics: Physics):
        """Step control"""
        current_time = self.iteration*self.timestep
        self._controller.step(
            iteration=self.iteration,
            time=current_time,
            timestep=self.timestep,
        )
        if self._controller.joints_names[ControlType.POSITION]:
            joints_positions = self._controller.positions(
                iteration=self.iteration,
                time=current_time,
                timestep=self.timestep,
            )
            physics.data.ctrl[self.maps['ctrl']['pos']] = [
                joints_positions[joint]
                for joint
                in self._controller.joints_names[ControlType.POSITION]
            ]
        if self._controller.joints_names[ControlType.TORQUE]:
            joints_torques = self._controller.torques(
                iteration=self.iteration,
                time=current_time,
                timestep=self.timestep,
            )
            torques = self.units.torques
            physics.data.ctrl[self.maps['ctrl']['trq']] = [
                joints_torques[joint]*torques
                for joint
                in self._controller.joints_names[ControlType.TORQUE]
            ]

    def after_step(self, physics: Physics):
        """Operations after physics step"""

        # Checks
        self.iteration += 1
        assert self.iteration <= self.n_iterations

        # Simulation complete
        if self.iteration == self.n_iterations:
            pylog.info('Simulation complete')
            if self._app is not None and not self._restart:
                self._app.close()
            else:
                pylog.info('Simulation can be restarted')

        # Callbacks
        for callback in self._callbacks:
            callback.after_step(task=self, physics=physics)

    def action_spec(self, physics: Physics):
        """Action specifications"""
        specs = []
        for callback in self._callbacks:
            spec = callback.action_spec(task=self, physics=physics)
            if spec is not None:
                specs += spec
        return specs

    def step_spec(self, physics: Physics):
        """Timestep specifications"""
        for callback in self._callbacks:
            callback.step_spec(task=self, physics=physics)

    def get_observation(self, physics: Physics):
        """Environment observation"""
        for callback in self._callbacks:
            callback.get_observation(task=self, physics=physics)

    def get_reward(self, physics: Physics):
        """Reward"""
        reward = 0
        for callback in self._callbacks:
            callback_reward = callback.get_reward(task=self, physics=physics)
            if callback_reward is not None:
                reward += callback_reward
        return reward

    def get_termination(self, physics: Physics):
        """Return final discount if episode should end, else None"""
        terminate = None
        for callback in self._callbacks:
            if callback.get_termination(task=self, physics=physics):
                terminate = 1
        if self.iteration >= self.n_iterations:
            terminate = 1
        return terminate

    def observation_spec(self, physics: Physics):
        """Observation specifications"""
        for callback in self._callbacks:
            callback.observation_spec(task=self, physics=physics)


class TaskCallback:
    """Task callback"""

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        """Initialize episode"""

    def before_step(self, task: ExperimentTask, action, physics: Physics):
        """Before step"""

    def after_step(self, task: ExperimentTask, physics: Physics):
        """After step"""

    def action_spec(self, task: ExperimentTask, physics: Physics):
        """Action specifications"""

    def step_spec(self, task: ExperimentTask, physics: Physics):
        """Timestep specifications"""

    def get_observation(self, task: ExperimentTask, physics: Physics):
        """Environment observation"""

    def get_reward(self, task: ExperimentTask, physics: Physics):
        """Reward"""

    def get_termination(self, task: ExperimentTask, physics: Physics):
        """Return final discount if episode should end, else None"""

    def observation_spec(self, task: ExperimentTask, physics: Physics):
        """Observation specifications"""
