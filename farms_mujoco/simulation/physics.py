"""Physics"""

import numpy as np

from farms_core import pylog
# pylint: disable=no-name-in-module
from farms_core.sensors.sensor_convention import sc
from ..sensors.sensors import cycontacts2data, cymusclesensors2data


def links_data(physics, sensor_maps):
    """Read links data"""
    return [
        physics.data.sensordata[
            sensor_maps[identifier]['indices'].flatten()
        ].reshape(
            [len(sensor_maps[identifier]['names']), n_c]
        )
        if len(sensor_maps[identifier]['names']) > 0
        else []
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
        if len(sensor_maps[identifier]['names']) > 0
        else []
        for identifier, n_c in [
                ['jointpos', 1],
                ['jointvel', 1],
                ['actuatorfrc_position', 1],
                ['actuatorfrc_velocity', 1],
                ['actuatorfrc_torque', 1],
        ]
    ]


def row2index(row, name, single=False):
    """Row to index"""
    identifier = row.convert_key_item(name)
    if isinstance(identifier, slice):
        indices = [
            np.arange(id_slice.start, id_slice.stop, id_slice.step)
            for id_slice in [identifier]
        ][0]
        return indices[0] if single else indices
    return identifier


def get_sensor_maps(physics, verbose=True):
    """Sensors information"""
    sensors_row = physics.named.data.sensordata.axes.row
    sensors_names = sensors_row.names
    if verbose:
        pylog.info('Sensors data:\n%s', physics.named.data.sensordata)
    sensors = [
        # Links
        'framepos', 'framequat', 'framelinvel', 'frameangvel',
        # Joints
        'jointpos', 'jointvel', 'jointlimitfrc',
        'force', 'torque',
        # Joints control
        'actuatorfrc_position', 'actuatorfrc_velocity', 'actuatorfrc_torque',
        # Muscles
        'musclefrc', 'tendonpos', 'tendonvel',
        'musclefiberlen', 'musclefibervel',
        'musclepenn', 'muscleactivefrc', 'musclepassivefrc',
        'muscleIa', 'muscleII', 'muscleIb',
        # Contacts
        'touch',
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
            row2index(row=sensors_row, name=name)
            for name in sensor_info['names']
        ])

    for sensor_info in sensor_maps.values():
        sensor_info['indices'] = np.array([
            [
                np.arange(id_slice.start, id_slice.stop, id_slice.step)
                for id_slice in [sensors_row.convert_key_item(name)]
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
                    ['torques (torque)', 'actuatorfrc_torque'],
                    ['limits', 'jointlimitfrc'],
                    ['forces (total)', 'force'],
                    ['torques (total)', 'torque'],
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
        # physics.data.xfrc_applied[:] = 0
        pylog.info(physics.named.data.xfrc_applied)

    return sensor_maps


def get_physics2data_maps(physics, sensor_data, sensor_maps):
    """Sensor to data maps"""

    # Names from data
    links_names = sensor_data.links.names
    joints_names = sensor_data.joints.names
    muscles_names = sensor_data.muscles.names

    # Links from physics
    xpos_row = physics.named.data.xpos.axes.row
    sensor_maps['xpos2data'] = np.array([
        row2index(row=xpos_row, name=link_name)
        for link_name in links_names
    ])
    xquat_row = physics.named.data.xquat.axes.row
    sensor_maps['xquat2data'] = np.array([
        row2index(row=xquat_row, name=link_name)
        for link_name in links_names
    ])
    xipos_row = physics.named.data.xipos.axes.row
    sensor_maps['xipos2data'] = np.array([
        row2index(row=xipos_row, name=link_name)
        for link_name in links_names
    ])
    cvel_row = physics.named.data.cvel.axes.row
    sensor_maps['cvel2data'] = np.array([
        row2index(row=cvel_row, name=link_name)
        for link_name in links_names
    ])

    # Joints from physics
    qpos_row = physics.named.data.qpos.axes.row
    sensor_maps['qpos2data'] = np.array([
        row2index(row=qpos_row, name=joint_name, single=True)
        for joint_name in joints_names
    ])
    qvel_row = physics.named.data.qvel.axes.row
    sensor_maps['qvel2data'] = np.array([
        row2index(row=qvel_row, name=joint_name, single=True)
        for joint_name in joints_names
    ])

    # Links - sensors
    for identifier in ['framepos', 'framequat', 'framelinvel', 'frameangvel']:
        sensor_maps[f'{identifier}2data'] = np.array([
            sensor_maps[identifier]['indices'][
                sensor_maps[identifier]['names'].index(
                    f'{identifier}_{link_name}'
                )
            ]
            for link_name in links_names
        ]) if all(
            f'{identifier}_{link_name}' in sensor_maps[identifier]['names']
            for link_name in links_names
        ) else []
    if len(sensor_maps['framequat2data']) > 0:
        sensor_maps['framequat2data'][:, :] = (
            sensor_maps['framequat2data'][:, [1, 2, 3, 0]]
        )

    # Joints - sensors
    for identifier in [
            'jointpos', 'jointvel', 'jointlimitfrc',
            'actuatorfrc_position',
            'actuatorfrc_velocity',
            'actuatorfrc_torque',
    ]:
        sensor_maps[f'{identifier}2data'] = np.array([
            sensor_maps[identifier]['indices'][
                sensor_maps[identifier]['names'].index(
                    f'{identifier}_{joint_name}'
                )
            ][0]
            for joint_name in joints_names
        ]) if all(
            f'{identifier}_{joint_name}' in sensor_maps[identifier]['names']
            for joint_name in joints_names
        ) else []
    forces_row = physics.named.data.sensordata.axes.row
    sensor_maps['force2data'] = np.array([
        row2index(row=forces_row, name=f'force_{joint_name}')
        for joint_name in joints_names
    ])
    torques_row = physics.named.data.sensordata.axes.row
    sensor_maps['torque2data'] = np.array([
        row2index(row=torques_row, name=f'torque_{joint_name}')
        for joint_name in joints_names
    ])

    # Muscles - sensors
    for identifier in [
            'musclefrc',
            'musclefiberlen', 'musclefibervel', 'musclepenn',
            'muscleactivefrc', 'musclepassivefrc',
            'muscleIa', 'muscleII', 'muscleIb'
    ]:
        sensor_maps[f'{identifier}2data'] = np.array([
            sensor_maps[identifier]['indices'][
                sensor_maps[identifier]['names'].index(
                    f'{identifier}_{muscle_name}'
                )
            ][0]
            for muscle_name in muscles_names
        ]) if all(
            f'{identifier}_{muscle_name}' in sensor_maps[identifier]['names']
            for muscle_name in muscles_names
        ) else []

    tendonlen_row = physics.named.data.ten_length.axes.row
    sensor_maps['tendonpos2data'] = np.array([
        row2index(row=tendonlen_row, name=f'{muscle_name}')
        for muscle_name in muscles_names
    ])
    tendonvel_row = physics.named.data.ten_velocity.axes.row
    sensor_maps['tendonvel2data'] = np.array([
        row2index(row=tendonvel_row, name=f'{muscle_name}')
        for muscle_name in muscles_names
    ])
    # Muscle sensors
    sensor_maps['musclesensors2data'] = np.array([
        [
            row2index(
                row=physics.named.data.ctrl.axes.row,
                name=f'{muscle_name}'
            ),
            row2index(
                row=physics.named.data.act.axes.row,
                name=f'{muscle_name}'
            ),
            row2index(
                row=physics.named.data.actuator_length.axes.row,
                name=f'{muscle_name}'
            ),
            row2index(
                row=physics.named.data.actuator_velocity.axes.row,
                name=f'{muscle_name}'
            ),
            row2index(
                row=physics.named.data.actuator_force.axes.row,
                name=f'{muscle_name}'
            ),
            row2index(
                row=physics.named.model.actuator_gainprm.axes.row,
                name=f'{muscle_name}'
            ),
            row2index(
                row=physics.named.model.actuator_user.axes.row,
                name=f'{muscle_name}'
            ),
        ]
        for muscle_name in muscles_names
    ])

    # Actuator - sensors
    actuator_momentrow = physics.named.data.actuator_moment.axes.row
    actuator_momentcol = physics.named.data.actuator_moment.axes.col
    actuator_moment_numrows = len(actuator_momentrow.names)
    actuator_moment_numcols = len(actuator_momentcol.names)
    for identifier in ['actuator_moment',]:
        sensor_maps[f'{identifier}2data'] = np.concatenate([
            row2index(
                actuator_momentrow, name=actuator_name
            )*actuator_moment_numcols + row2index(
                actuator_momentcol, name=joint_name
            )
            for actuator_name in actuator_momentrow.names
            for joint_name in actuator_momentcol.names
        ]).ravel()

    # Contacts
    contacts_pairs = sensor_data.contacts.names
    body_names = physics.named.model.body_pos.axes.row.names
    sensor_maps['geompair2data'] = {
        (geom_id, -1): contacts_pairs.index((body_names[body_id], ''))
        for geom_id, body_id in enumerate(physics.model.geom_bodyid)
        if (body_names[body_id], '') in contacts_pairs
    }
    sensor_maps['geompair2data'].update({
        (geom_id1, geom_id2): contacts_pairs.index(
            (body_names[body_id1], body_names[body_id2])
        )
        for geom_id1, body_id1 in enumerate(physics.model.geom_bodyid)
        for geom_id2, body_id2 in enumerate(physics.model.geom_bodyid)
        if (body_names[body_id1], body_names[body_id2]) in contacts_pairs
    })
    geompair2data_values = sensor_maps['geompair2data'].values()
    for pair_i, pair in enumerate(contacts_pairs):
        assert not isinstance(pair, str) and len(pair) == 2, (
            f'Contact "{pair}" should be a pair of strings'
        )
        assert pair_i in geompair2data_values, (
            f'Missing pair: {pair} ({body_names=})'
        )

    # External forces
    row = physics.named.data.xfrc_applied.axes.row
    sensor_maps['data2xfrc'] = np.array([
        row2index(row=row, name=name, single=True)
        for name in sensor_data.xfrc.names
    ])
    sensor_maps['datalinks2xfrc'] = np.array([
        row2index(row=row, name=name, single=True)
        for name in sensor_data.links.names
    ])


def physics_muscles_sensors2data(physics, iteration, data, sensor_maps, units):
    """ Sensor data collection for muscles """
    # tendon lengths
    data.sensors.muscles.array[
        iteration, :,
        sc.muscle_tendon_unit_length,
    ] = physics.data.ten_length[sensor_maps['tendonpos2data']]/units.meters
    # tendon velocities
    data.sensors.muscles.array[
        iteration, :,
        sc.muscle_tendon_unit_velocity,
    ] = physics.data.ten_velocity[sensor_maps['tendonvel2data']]/units.velocity
    data.sensors.muscles.array[
        iteration, :,
        sc.muscle_tendon_unit_force,
    ] = physics.data.sensordata[sensor_maps['musclefrc2data']]/units.newtons
    cymusclesensors2data(
        physics=physics,
        iteration=iteration,
        data=data.sensors.muscles,
        musclesensor2data=sensor_maps['musclesensors2data'],
        meters=units.meters,
        velocity=units.velocity,
        newtons=units.newtons
    )


def physicslinkssensors2data(physics, iteration, data, sensor_maps, units):
    """Sensors data collection"""
    data.sensors.links.array[
        iteration, :,
        sc.link_urdf_position_x:sc.link_urdf_position_z+1,
    ] = physics.data.sensordata[sensor_maps['framepos2data']]/units.meters
    data.sensors.links.array[
        iteration, :,
        sc.link_urdf_orientation_x:sc.link_urdf_orientation_w+1,
    ] = physics.data.sensordata[sensor_maps['framequat2data']]


def physicslinksvelsensors2data(physics, iteration, data, sensor_maps, units):
    """Sensors data collection"""
    data.sensors.links.array[
        iteration, :,
        sc.link_com_velocity_lin_x:sc.link_com_velocity_lin_z+1,
    ] = physics.data.sensordata[sensor_maps['framelinvel2data']]/units.velocity
    data.sensors.links.array[
        iteration, :,
        sc.link_com_velocity_ang_x:sc.link_com_velocity_ang_z+1,
    ] = (
        physics.data.sensordata[sensor_maps['frameangvel2data']]
    )/units.angular_velocity


def physicslinks2data(physics, iteration, data, sensor_maps, units):
    """Sensors data collection"""
    data.sensors.links.array[
        iteration, :,
        sc.link_urdf_position_x:sc.link_urdf_position_z+1,
    ] = physics.data.xpos[sensor_maps['xpos2data']]/units.meters
    data.sensors.links.array[
        iteration, :,
        sc.link_urdf_orientation_x:sc.link_urdf_orientation_w+1,
    ] = physics.data.xquat[sensor_maps['xquat2data']][:, [1, 2, 3, 0]]
    data.sensors.links.array[
        iteration, :,
        sc.link_com_position_x:sc.link_com_position_z+1,
    ] = physics.data.xipos[sensor_maps['xipos2data']]/units.meters
    data.sensors.links.array[
        iteration, :,
        sc.link_com_orientation_x:sc.link_com_orientation_w+1,
    ] = physics.data.xquat[sensor_maps['xquat2data']][:, [1, 2, 3, 0]]


def physicslinksvel2data(physics, iteration, data, sensor_maps, units):
    """Sensors data collection"""
    data.sensors.links.array[
        iteration, :,
        sc.link_com_velocity_lin_x:sc.link_com_velocity_lin_z+1,
    ] = physics.data.cvel[sensor_maps['cvel2data'], 3:]/units.velocity
    data.sensors.links.array[
        iteration, :,
        sc.link_com_velocity_ang_x:sc.link_com_velocity_ang_z+1,
    ] = physics.data.cvel[sensor_maps['cvel2data'], :3]/units.angular_velocity


def physicsjointssensors2data(physics, iteration, data, sensor_maps, units):
    """Sensors data collection"""
    # TODO: Check the units
    data.sensors.joints.array[iteration, :, sc.joint_limit_force] = (
        physics.data.sensordata[sensor_maps['jointlimitfrc2data']]
    )/units.torques
    # Total forces
    data.sensors.joints.array[iteration, :, sc.joint_force_x:sc.joint_force_z+1] = (
        physics.data.sensordata[sensor_maps['force2data']]
    )/units.newtons
    # Total torques
    data.sensors.joints.array[iteration, :, sc.joint_torque_x:sc.joint_torque_z+1] = (
        physics.data.sensordata[sensor_maps['torque2data']]
    )/units.torques


def physicsjoints2data(physics, iteration, data, sensor_maps, units):
    """Sensors data collection"""
    data.sensors.joints.array[iteration, :, sc.joint_position] = (
        physics.data.qpos[sensor_maps['qpos2data']]
    )
    data.sensors.joints.array[iteration, :, sc.joint_velocity] = (
        physics.data.qvel[sensor_maps['qvel2data']]
    )/units.angular_velocity


def physicsactuators2data(physics, iteration, data, sensor_maps, units):
    """Sensors data collection"""
    itorques = 1./units.torques
    if len(sensor_maps['actuatorfrc_position2data']) > 0:
        data.sensors.joints.array[iteration, :, sc.joint_torque] += (
            physics.data.sensordata[sensor_maps['actuatorfrc_position2data']]
        )*itorques
    if len(sensor_maps['actuatorfrc_velocity2data']) > 0:
        data.sensors.joints.array[iteration, :, sc.joint_torque] += (
            physics.data.sensordata[sensor_maps['actuatorfrc_velocity2data']]
        )*itorques
    if len(sensor_maps['actuatorfrc_torque2data']) > 0:
        data.sensors.joints.array[iteration, :, sc.joint_torque] += (
            physics.data.sensordata[sensor_maps['actuatorfrc_torque2data']]
        )*itorques


def physics2data(physics, iteration, data, maps, units, links_only=False):
    """Sensors data collection"""
    sensor_maps = maps['sensors']
    physicslinks2data(physics, iteration, data, sensor_maps, units)
    physicslinksvelsensors2data(physics, iteration, data, sensor_maps, units)
    if not links_only:
        physicsjointssensors2data(physics, iteration, data, sensor_maps, units)
        physicsjoints2data(physics, iteration, data, sensor_maps, units)
        physicsactuators2data(physics, iteration, data, sensor_maps, units)
        cycontacts2data(
            physics=physics,
            iteration=iteration,
            data=data.sensors.contacts,
            geompair2data=sensor_maps['geompair2data'],
            meters=units.meters,
            newtons=units.newtons,
        )
        if data.sensors.muscles.names:
            physics_muscles_sensors2data(physics, iteration, data, sensor_maps, units)
