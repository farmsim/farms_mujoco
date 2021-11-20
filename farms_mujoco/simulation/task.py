"""Task"""

from typing import Dict
from enum import IntEnum

import numpy as np

from dm_control.rl.control import Task

import farms_pylog as pylog
from farms_data.units import SimulationUnitScaling
from farms_data.amphibious.animat_data import ModelData

from .physics import (
    get_sensor_maps,
    get_physics2data_maps,
    physics2data,
)


class ControlType(IntEnum):
    """Control type"""
    POSITION = 0
    VELOCITY = 1
    TORQUE = 2


def duration2nit(duration: float, timestep: float) -> int:
    """Number of iterations from duration"""
    return int(duration/timestep)


class ExperimentTask(Task):
    """Defines a task in a `control.Environment`."""

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
        assert not kwargs, kwargs

    def set_app(self, app):
        """Set application"""
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

        # Data
        if self.data is None:
            self.data = ModelData.from_sensors_names(
                timestep=self.timestep,
                n_iterations=self.n_iterations,
                links=self.maps['xpos']['names'],
                joints=self.maps['qpos']['names'],
                # contacts=[],
                # hydrodynamics=[],
            )

        # Sensor maps
        self.maps['sensors'] = get_sensor_maps(physics)
        get_physics2data_maps(
            physics=physics,
            sensor_data=self.data.sensors,
            sensor_maps=self.maps['sensors'],
        )

        # Control
        if self._controller is not None:
            ctrl_names = np.array(physics.named.data.ctrl.axes.row.names)
            for joint in self._controller.joints_names[ControlType.POSITION]:
                assert f'actuator_position_{joint}' in ctrl_names, (
                    f'{joint} not in {ctrl_names}'
                )
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
            if self.animat_options is not None:
                animat_options = self.animat_options
                for jnt_opts in animat_options.control.joints:
                    jnt_name = jnt_opts['joint_name']
                    if ControlType.POSITION not in jnt_opts.control_types:
                        for act_type in ('pos', 'vel'):
                            if act_type in jntname2actid[jnt_name]:
                                physics.named.model.actuator_forcelimited[
                                    jntname2actid[jnt_name][act_type]
                                ] = 1
                                physics.named.model.actuator_forcerange[
                                    jntname2actid[jnt_name][act_type]
                                ] = [0, 0]

    def before_step(self, action, physics):
        """Operations before physics step"""

        # Checks
        assert self.iteration < self.n_iterations

        # Sensors
        physics2data(physics, self.iteration, self.data, self.maps, self._units)

        # # Print contacts
        # if 2 < physics.time() < 2.1:
        #     print_contacts(physics, self.maps['geoms']['names'])

        # # Set external force
        # if 3 < physics.time() < 4:
        #     index = np.argwhere(
        #         np.array(self.maps['xfrc']['names']) == self.base_link
        #     )[0, 0]
        #     physics.data.xfrc_applied[index, 2] = self.external_force
        # elif 2.9 < physics.time() < 3 or 4 < physics.time() < 4.1:
        #     physics.data.xfrc_applied[:] = 0  # No interaction

        # Control
        if self._controller is not None:
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
            else:
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
