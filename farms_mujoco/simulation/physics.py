"""Physics"""

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


def print_contacts(physics, geoms_names):
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
        physics.data.sensordata[maps['sensors'][identifier]['indices']].reshape(
            [len(maps['sensors'][identifier]['names']), n_c]
        )
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
        physics.data.sensordata[maps['sensors'][identifier]['indices']].reshape(
            [len(maps['joints']['names']), n_c]
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
