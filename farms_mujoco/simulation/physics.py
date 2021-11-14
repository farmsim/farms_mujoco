"""Physics"""

import numpy as np

import farms_pylog as pylog
# pylint: disable=no-name-in-module
from farms_data.sensors.sensor_convention import sc


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


def print_contacts(physics, geoms_names):
    """Print contacts"""
    contacts = collect_contacts(physics)
    if contacts:
        pylog.info('\n'.join([
            f'({geoms_names[geoms[0]]}, {geoms_names[geoms[1]]}): {force}'
            for geoms, force in contacts.items()
        ]))


def links_data(physics, sensor_maps):
    """Read links data"""
    return [
        physics.data.sensordata[
            sensor_maps[identifier]['indices'].flatten()
        ].reshape(
            [len(sensor_maps[identifier]['names']), n_c]
        )
        for identifier, n_c in [
                ['framepos', 3],
                ['framequat', 4],
                ['framelinvel', 3],
                ['frameangvel', 3],
        ]
    ]


def joints_data(physics, sensor_maps):
    """Read joints data"""
    return [
        physics.data.sensordata[
            sensor_maps[identifier]['indices'].flatten()
        ].reshape(
            [len(sensor_maps[identifier]['names']), n_c]
            if n_c > 1
            else len(sensor_maps[identifier]['names'])
        )
        for identifier, n_c in [
                ['jointpos', 1],
                ['jointvel', 1],
                ['actuatorfrc_position', 1],
                ['actuatorfrc_velocity', 1],
        ]
    ]


def get_sensor_maps(physics, verbose=True):
    """Sensors information"""
    sensors_row = physics.named.data.sensordata.axes.row
    sensors_names = sensors_row.names
    pylog.info('Sensors data:\n%s', physics.named.data.sensordata)
    sensors = [
        'framepos', 'framequat', 'framelinvel', 'frameangvel',  # Links
        'jointpos', 'jointvel',  # Joints
        'actuatorfrc_position', 'actuatorfrc_velocity',  # Joints control
        'touch',  # Contacts
    ]
    sensor_maps = {
        sensor: {
            'names': [
                name
                for name in sensors_names
                if name.startswith(sensor)
            ],
        }
        for sensor in sensors
    }
    for sensor_info in sensor_maps.values():
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
        ])

    if verbose:

        # Links sensors
        for (name, identifier), data in zip(
                [
                    ['positions', 'framepos'],
                    ['orientations', 'framequat'],
                    ['linear velocities', 'framelinvel'],
                    ['angular velocities', 'frameangvel'],
                ],
                links_data(physics, sensor_maps),
        ):
            pylog.info(
                'Links initial %s:\n%s',
                name,
                '\n'.join([
                    f'{link_i} - {name}: {value}'
                    for link_i, (name, value) in enumerate(zip(
                            sensor_maps[identifier]['names'],
                            data,
                    ))
                ])
            )

        # Joints sensors
        for (name, identifier), data in zip(
                [
                    ['positions', 'jointpos'],
                    ['velocities', 'jointvel'],
                    ['torques (position)', 'actuatorfrc_position'],
                    ['torques (velocity)', 'actuatorfrc_velocity'],
                ],
                joints_data(physics, sensor_maps),
        ):
            pylog.info(
                'Joints initial %s:\n%s',
                name,
                '\n'.join([
                    f'{joint_i} - {name}: {value}'
                    for joint_i, (name, value) in enumerate(zip(
                            sensor_maps[identifier]['names'],
                            data,
                    ))
                ])
            )

        # Contacts sensors
        for name, identifier in [
                ['contacts', 'touch'],
        ]:
            if len(sensor_maps[identifier]['indices']) == 0:
                continue
            pylog.info(
                'Geometry initial %s:\n%s',
                name,
                '\n'.join([
                    f'{name}: {value}'
                    for name, value in zip(
                            sensor_maps[identifier]['names'],
                            physics.data.sensordata[
                                sensor_maps[identifier]['indices'].flatten()
                            ],
                    )
                ])
            )

        # External forces in world frame
        physics.data.xfrc_applied[:] = 0
        pylog.info(physics.named.data.xfrc_applied)

    return sensor_maps


def get_physics2data_maps(physics, sensor_data, sensor_maps):
    """Sensor to data maps"""

    # Names from data
    links_names = sensor_data.links.names
    joints_names = sensor_data.joints.names

    # Links from physics
    xpos_names = physics.named.data.xpos.axes.row.names
    sensor_maps['xpos2data'] = np.array([
        xpos_names.index(link_name)
        for link_name in links_names
    ])
    xquat_names = physics.named.data.xquat.axes.row.names
    sensor_maps['xquat2data'] = np.array([
        xquat_names.index(link_name)
        for link_name in links_names
    ])

    # # Joints from physics
    # qpos_names = physics.named.data.xpos.axes.row.names
    # sensor_maps['qpos2data'] = np.array([
    #     qpos_names.index(joint_name)
    #     for joint_name in joints_names
    # ])
    # qvel_names = physics.named.data.qvel.axes.row.names
    # sensor_maps['qvel2data'] = np.array([
    #     qvel_names.index(joint_name)
    #     for joint_name in joints_names
    # ])

    # Links - sensors
    for identifier in ['framepos', 'framequat', 'framelinvel', 'frameangvel']:
        sensor_maps[f'{identifier}2data'] = np.array([
            sensor_maps[identifier]['indices'][
                sensor_maps[identifier]['names'].index(
                    f'{identifier}_{link_name}'
                )
            ]
            for link_name in links_names
        ])
    sensor_maps['framequat2data'][:, :] = (
        sensor_maps['framequat2data'][:, [1, 2, 3, 0]]
    )

    # Joints - sensors
    for identifier in [
            'jointpos', 'jointvel',
            'actuatorfrc_position', 'actuatorfrc_velocity',
    ]:
        sensor_maps[f'{identifier}2data'] = np.array([
            sensor_maps[identifier]['indices'][
                sensor_maps[identifier]['names'].index(
                    f'{identifier}_{joint_name}'
                )
            ][0]
            for joint_name in joints_names
        ])


def physics2data(physics, iteration, data, maps):
    """Sensors data collection"""

    sensor_maps = maps['sensors']

    # Links
    # data.sensors.links.array[iteration, :,
    #     sc.link_urdf_position_x:sc.link_urdf_position_z+1,
    # ] = physics.data.sensordata[sensor_maps['framepos2data']]
    # data.sensors.links.array[iteration, :,
    #     sc.link_urdf_orientation_x:sc.link_urdf_orientation_w+1,
    # ] = physics.data.sensordata[sensor_maps['framequat2data']]
    data.sensors.links.array[iteration, :,
        sc.link_urdf_position_x:sc.link_urdf_position_z+1,
    ] = physics.data.xpos[sensor_maps['xpos2data']]
    data.sensors.links.array[iteration, :,
        sc.link_urdf_orientation_x:sc.link_urdf_orientation_w+1,
    ] = physics.data.xquat[sensor_maps['xquat2data']][:, [1, 2, 3, 0]]
    data.sensors.links.array[iteration, :,
            sc.link_com_velocity_lin_x:sc.link_com_velocity_lin_z+1,
    ] = physics.data.sensordata[sensor_maps['framelinvel2data']]
    data.sensors.links.array[iteration, :,
            sc.link_com_velocity_ang_x:sc.link_com_velocity_ang_z+1,
    ] = physics.data.sensordata[sensor_maps['frameangvel2data']]

    # Joints
    data.sensors.joints.array[iteration, :, sc.joint_position] = (
        physics.data.sensordata[sensor_maps['jointpos2data']]
    )
    data.sensors.joints.array[iteration, :, sc.joint_velocity] = (
        physics.data.sensordata[sensor_maps['jointvel2data']]
    )
    # data.sensors.joints.array[iteration, :, sc.joint_position] = (
    #     physics.data.qpos[sensor_maps['qpos2data']]
    # )
    # data.sensors.joints.array[iteration, :, sc.joint_velocity] = (
    #     physics.data.qvel[sensor_maps['qvel2data']]
    # )
    data.sensors.joints.array[iteration, :, sc.joint_torque] = (
        physics.data.sensordata[sensor_maps['actuatorfrc_position2data']]
        + physics.data.sensordata[sensor_maps['actuatorfrc_velocity2data']]
    )
