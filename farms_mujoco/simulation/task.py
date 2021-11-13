"""Task"""

from typing import Dict
from enum import IntEnum

import numpy as np
import matplotlib.pyplot as plt

from dm_control.rl.control import Task

import farms_pylog as pylog
from farms_data.sensors.sensor_convention import sc
from farms_data.amphibious.animat_data import ModelData

from .physics import physics2data, links_data, joints_data, print_contacts


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

    def __init__(self, base_link, duration, timestep, **kwargs):
        super().__init__()
        self._app = None
        self.iteration: int = 0
        self.duration: float = duration
        self.timestep: float = timestep
        self.n_iterations: int = duration2nit(duration, timestep)
        self.data: ModelData = kwargs.pop('data', None)
        self.base_link: str = base_link  # Link which to apply external force
        self.maps: Dict = {
            'links': {}, 'joints': {},
            'sensors': {}, 'xfrc': {}, 'geoms': {},
            'ctrl': {}
        }
        self.external_force: float = kwargs.pop('external_force', 0.2)
        self._restart: bool = kwargs.pop('restart', True)
        self._plot: bool = kwargs.pop('plot', False)
        self._controller = kwargs.pop('controller', None)
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
        if self._controller is not None:
            for name in self._controller.joints_names[ControlType.POSITION]:
                pass

        # Initialise iterations
        self.iteration = 0

        # Links indices
        self.maps['links']['names'] = list(
            physics.named.data.xpos.axes.row.names
        )
        if 'world' in self.maps['links']['names']:
            self.maps['links']['names'].remove('world')

        # Joints indices
        self.maps['joints']['names'] = list(
            physics.named.data.qpos.axes.row.names
        )
        if f'root_{self.base_link}' in self.maps['joints']['names']:
            self.maps['joints']['names'].remove(f'root_{self.base_link}')

        # External forces indices
        self.maps['xfrc']['names'] = (
            physics.named.data.xfrc_applied.axes.row.names
        )

        # Geoms indices
        self.maps['geoms']['names'] = (
            physics.named.data.geom_xpos.axes.row.names
        )

        # Sensors indices
        sensors_row = physics.named.data.sensordata.axes.row
        sensors_names = sensors_row.names
        pylog.info('Sensors data:\n%s', physics.named.data.sensordata)
        sensors = [
            'framepos', 'framequat', 'framelinvel', 'frameangvel',  # Links
            'jointpos', 'jointvel', 'actuatorfrc',  # Joints
            'touch',  # Contacts
        ]
        self.maps['sensors'] = {
            sensor: {
                'names': [
                    name
                    for name in sensors_names
                    if name.startswith(sensor)
                ],
            }
            for sensor in sensors
        }
        for sensor_info in self.maps['sensors'].values():
            sensor_info['indices'] = np.array([
                [
                    np.arange(
                        indices_slice.start,
                        indices_slice.stop,
                        indices_slice.step,
                    )
                    for indices_slice in [sensors_row.convert_key_item(name)]
                ][0]
                for name in sensor_info['names']
            ]).flatten()

        # Links sensors
        for (name, identifier), data in zip(
                [
                    ['positions', 'framepos'],
                    ['orientations', 'framequat'],
                    ['linear velocities', 'framelinvel'],
                    ['angular velocities', 'frameangvel'],
                ],
                links_data(physics, self.maps),
        ):
            pylog.info(
                'Links initial %s:\n%s',
                name,
                '\n'.join([
                    f'{link_i} - {name}: {value}'
                    for link_i, (name, value) in enumerate(zip(
                            self.maps['sensors'][identifier]['names'],
                            data,
                    ))
                ])
            )

        # Joints sensors
        for (name, identifier), data in zip(
                [
                    ['positions', 'jointpos'],
                    ['velocities', 'jointvel'],
                    ['torques', 'actuatorfrc'],
                ],
                joints_data(physics, self.maps),
        ):
            pylog.info(
                'Joints initial %s:\n%s',
                name,
                '\n'.join([
                    f'{joint_i} - {name}: {value}'
                    for joint_i, (name, value) in enumerate(zip(
                            self.maps['sensors'][identifier]['names'],
                            data,
                    ))
                ])
            )

        # Contacts sensors
        for name, identifier in [
                ['contacts', 'touch'],
        ]:
            if len(self.maps['sensors'][identifier]['indices']) == 0:
                continue
            pylog.info(
                'Geometry initial %s:\n%s',
                name,
                '\n'.join([
                    f'{name}: {value}'
                    for name, value in zip(
                            self.maps['sensors'][identifier]['names'],
                            physics.data.sensordata[
                                self.maps['sensors'][identifier]['indices']
                            ],
                    )
                ])
            )

        # External forces in world frame
        physics.data.xfrc_applied[:] = 0
        pylog.info(physics.named.data.xfrc_applied)

        # Data
        if self.data is None:
            self.data = ModelData.from_sensors_names(
                timestep=self.timestep,
                n_iterations=self.n_iterations,
                links=self.maps['links']['names'],
                joints=self.maps['joints']['names'],
                # contacts=[],
                # hydrodynamics=[],
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

    def before_step(self, action, physics):
        """Operations before physics step"""

        # Checks
        assert self.iteration < self.n_iterations

        # Sensors
        physics2data(physics, self.iteration, self.data, self.maps)

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
                    for joint in self._controller.joints_names[ControlType.POSITION]
                ]


    def after_step(self, physics):
        """Operations after physics step"""

        # Checks
        self.iteration += 1
        assert self.iteration <= self.n_iterations

        # Simulation complete
        if self.iteration == self.n_iterations:
            pylog.info('Simulation complete')
            if self._plot:
                times = np.arange(0, self.duration, self.timestep)
                self.data.plot_sensors(times=times)
                plt.show()
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
