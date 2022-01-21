"""Task"""

from typing import Dict
from enum import IntEnum

import numpy as np

from dm_control import viewer
from dm_control.rl.control import Task
from dm_control.mujoco.wrapper import mjbindings

import farms_pylog as pylog
from farms_data.model.control import ControlType
from farms_data.units import SimulationUnitScaling
from farms_data.sensors.sensor_convention import sc
from farms_data.amphibious.animat_data import ModelData

from ..swimming.drag import SwimmingHandler
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

    def __init__(self, base_link, n_iterations, timestep, **kwargs):
        super().__init__()
        self._app = None
        self.iteration: int = 0
        self.timestep: float = timestep
        self.n_iterations: int = n_iterations
        self.base_link: str = base_link
        self.data: ModelData = kwargs.pop('data', None)
        self._controller = kwargs.pop('controller', None)
        self.animat_options = kwargs.pop('animat_options', None)
        self.maps: Dict = {
            'sensors': {}, 'ctrl': {},
            'xpos': {}, 'qpos': {}, 'xfrc': {}, 'geoms': {},
            'links': {}, 'joints': {}, 'contacts': {}, 'hydrodynamics': {},
        }
        self.external_force: float = kwargs.pop('external_force', 0.2)
        self._restart: bool = kwargs.pop('restart', True)
        self._plot: bool = kwargs.pop('plot', False)
        self._save: str = kwargs.pop('save', '')
        self._units = kwargs.pop('units', SimulationUnitScaling)
        self._swimming_handler: SwimmingHandler = None
        self._hfield = kwargs.pop('hfield', None)
        assert not kwargs, kwargs

    def set_app(self, app: viewer.application.Application):
        """Set application"""
        assert isinstance(app, viewer.application.Application)
        self._app = app

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode"""

        # Checks
        if self._restart:
            assert self._app is not None, (
                'Simulation can not be restarted without application interface'
            )

        # Initialise iterations
        self.iteration = 0

        # Initialise terrain
        if self._hfield is not None:
            data = self._hfield['data']
            hfield = self._hfield['asset']
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
            self.initialize_data()
        self.initialize_sensors(physics)

        # Control
        if self._controller is not None:
            self.initialize_control(physics)

        # Hydrodynamics
        if self.animat_options.physics.drag or self.animat_options.physics.sph:
            self._swimming_handler = SwimmingHandler(
                data=self.data,
                animat_options=self.animat_options,
                units=self._units,
                physics=physics,
            )

        # Initialize joints
        for joint in self.animat_options.morphology.joints:
            assert joint.name in self.maps['qpos']['names']
            index = self.maps['qpos']['names'].index(joint.name)+6
            physics.data.qpos[index] = joint.initial_position
            physics.data.qvel[index-1] = joint.initial_velocity

    def before_step(self, action, physics):
        """Operations before physics step"""

        # Checks
        assert self.iteration < self.n_iterations

        # Sensors
        physics2data(
            physics=physics,
            iteration=self.iteration,
            data=self.data,
            maps=self.maps,
            units=self._units,
        )

        # Hydrodynamics
        if self._swimming_handler is not None:
            self.step_hydrodynamics(physics)

        # Control
        if self._controller is not None:
            self.step_control(physics)

    def initialize_maps(self, physics):
        """Initialise data"""
        # Links indices
        self.maps['xpos']['names'] = list(
            physics.named.data.xpos.axes.row.names
        )

        # Joints indices
        self.maps['qpos']['names'] = list(
            physics.named.data.qpos.axes.row.names
        )

        # External forces indices
        self.maps['xfrc']['names'] = (
            physics.named.data.xfrc_applied.axes.row.names
        )

        # Geoms indices
        self.maps['geoms']['names'] = (
            physics.named.data.geom_xpos.axes.row.names
        )

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

    def initialize_sensors(self, physics):
        """Initialise sensors"""
        self.maps['sensors'] = get_sensor_maps(physics)
        get_physics2data_maps(
            physics=physics,
            sensor_data=self.data.sensors,
            sensor_maps=self.maps['sensors'],
        )

    def initialize_control(self, physics):
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

    def step_hydrodynamics(self, physics):
        """Step hydrodynamics"""
        self._swimming_handler.step(self.iteration)
        # physics.data.xfrc_applied[:, :] = 0  # Reset all forces
        indices = self.maps['sensors']['data2xfrc2']
        physics.data.xfrc_applied[indices, :] = (
            self.data.sensors.hydrodynamics.array[
                self.iteration, :,
                sc.hydrodynamics_force_x:sc.hydrodynamics_torque_z+1,
            ]
        )
        for force_i, (rotation_mat, force_local) in enumerate(zip(
                physics.data.xmat[indices],
                physics.data.xfrc_applied[indices],
        )):
            physics.data.xfrc_applied[indices[force_i]] = (
                rotation_mat.reshape([3, 3])  # Local to global frame
                @ force_local.reshape([3, 2], order='F')
            ).flatten(order='F')
        physics.data.xfrc_applied[indices, :3] *= self._units.newtons
        physics.data.xfrc_applied[indices, 3:] *= self._units.torques

    def step_control(self, physics):
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
            torques = self._units.torques
            physics.data.ctrl[self.maps['ctrl']['trq']] = [
                joints_torques[joint]*torques
                for joint
                in self._controller.joints_names[ControlType.TORQUE]
            ]

    def after_step(self, physics):
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

    def action_spec(self, physics):
        """Action specifications"""
        return []

    def step_spec(self, physics):
        """Timestep specifications"""

    def get_observation(self, physics):
        """Environment observation"""

    def get_reward(self, physics):
        """Reward"""
        return 0

    def get_termination(self, physics):
        """Return final discount if episode should end, else None"""
        return 1 if self.iteration >= self.n_iterations else None

    def observation_spec(self, physics):
        """Observation specifications"""
