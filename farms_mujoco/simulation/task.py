"""Task"""

from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from dm_control.rl.control import Task

import farms_pylog as pylog
from farms_data.sensors.sensor_convention import sc
from farms_data.amphibious.animat_data import ModelData


def collect_contacts(physics):
    """Collect contacts"""
    contacts = {}
    for contact_i, contact in enumerate(physics.data.contact):
        if contact.dist < contact.includemargin:
            forcetorque = physics.data.contact_force(contact_i)
            reaction = forcetorque[0, 0]*contact.frame[0:3]
            friction1 = forcetorque[0, 1]*contact.frame[3:6]
            friction2 = forcetorque[0, 2]*contact.frame[6:9]
            contacts[(contact.geom1, contact.geom2)] = (
                reaction + friction1 + friction2
                + contacts.get((contact.geom1, contact.geom2), 0.)
            )
    return contacts


def print_contacts(geoms_names, physics):
    """Print contacts"""
    contacts = collect_contacts(physics)
    if contacts:
        pylog.info('\n'.join([
            f'({geoms_names[geoms[0]]}, {geoms_names[geoms[1]]}): {force}'
            for geoms, force in contacts.items()
        ]))


def links_data(physics, maps):
    """Read links data"""
    return [
        physics.data.sensordata[
            maps['sensors'][identifier]['indices']
        ].reshape([
            len(maps['sensors'][identifier]['names']),
            n_c,
        ])
        for identifier, n_c in [
                ['framepos', 3],
                ['framequat', 4],
                ['framelinvel', 3],
                ['frameangvel', 3],
        ]
    ]


def joints_data(physics, maps):
    """Read joints data"""
    return [
        physics.data.sensordata[
            maps['sensors'][identifier]['indices']
        ].reshape(
            [
                len(maps['joints']['names']),
                n_c,
            ]
            if n_c > 1
            else len(maps['joints']['names'])
        )
        for identifier, n_c in [
                ['jointpos', 1],
                ['jointvel', 1],
                ['actuatorfrc', 2],
        ]
    ]


def physics2data(physics, iteration, data, maps):
    """Sensors data collection"""

    # Links
    framepos, framequat, framelinvel, frameangvel = links_data(physics, maps)
    data.sensors.links.array[iteration, :,
        sc.link_urdf_position_x:sc.link_urdf_position_z+1,
    ] = framepos
    data.sensors.links.array[iteration, :,
            sc.link_urdf_orientation_x:sc.link_urdf_orientation_w+1,
    ] = framequat[:, [3, 0, 1, 2]]
    data.sensors.links.array[iteration, :,
            sc.link_com_velocity_lin_x:sc.link_com_velocity_lin_z+1,
    ] = framelinvel
    data.sensors.links.array[iteration, :,
            sc.link_com_velocity_ang_x:sc.link_com_velocity_ang_z+1,
    ] = frameangvel

    # Joints
    jointpos, jointvel, actuatorfrc = joints_data(physics, maps)
    data.sensors.joints.array[iteration, :, sc.joint_position] = (
        jointpos
    )
    data.sensors.joints.array[iteration, :, sc.joint_velocity] = (
        jointvel
    )
    data.sensors.joints.array[iteration, :, sc.joint_torque] = np.sum(
        actuatorfrc,
        axis=1,
    )


class ExperimentTask(Task):
    """Defines a task in a `control.Environment`."""

    def __init__(self, base_link, duration, timestep, **kwargs):
        super().__init__()
        self._app = None
        self.iteration: int = 0
        self.duration: float = duration
        self.timestep: float = timestep
        self.n_iterations: int = int(duration/timestep)
        self.data: ModelData = kwargs.pop('data', None)
        self.base_link: str = base_link  # Link which to apply external force
        self.maps: Dict = {
            'links': {},
            'joints': {},
            'sensors': {},
            'xfrc': {},
            'geoms': {},
        }
        self.external_force: float = kwargs.pop('external_force', 0.2)
        self._restart = kwargs.pop('restart', True)
        self._plot = kwargs.pop('plot', False)
        assert not kwargs, kwargs

    def set_app(self, app):
        """Set application"""
        self._app = app

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode"""

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

    def before_step(self, action, physics):
        """Operations before physics step"""

        # Sensors
        physics2data(physics, self.iteration, self.data, self.maps)

        # Print contacts
        if 2 < physics.time() < 2.1:
            print_contacts(self.maps['geoms']['names'], physics)

        # Set external force
        if 3 < physics.time() < 4:
            index = np.argwhere(
                np.array(self.maps['xfrc']['names']) == self.base_link
            )[0, 0]
            physics.data.xfrc_applied[index, 2] = self.external_force
        elif 2.9 < physics.time() < 3 or 4 < physics.time() < 4.1:
            physics.data.xfrc_applied[:] = 0  # No interaction

        # Control
        freq = 1.0
        amp = 0.1
        controls = [
            [amp*np.sin(2*np.pi*freq*physics.time()), 0.0]
            for i in range(int(physics.model.nu/2))
        ]
        physics.set_control(np.array(controls).flatten())


    def after_step(self, physics):
        """Operations after physics step"""
        self.iteration += 1
        assert self.iteration <= self.n_iterations
        if self.iteration == self.n_iterations:
            pylog.info('Simulation complete')
            if self._plot:
                times = np.arange(0, self.duration, self.timestep)
                self.data.plot_sensors(times=times)
                plt.show()
            if self._app is not None and not self._restart:
                self._app.close()

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
        return None

    def observation_spec(self, physics):
        """Observation specifications"""