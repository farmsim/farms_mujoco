"""Swimming"""

import numpy as np
import pybullet

import farms_pylog as pylog


def link_swimming_info(data_gps, iteration, sensor_i):
    """Link swimming information"""

    # Orientations
    ori_urdf = data_gps.urdf_orientation(iteration, sensor_i)
    urdf2global = [0, 0, 0], ori_urdf
    ori_com = data_gps.com_orientation(iteration, sensor_i)
    global2com = pybullet.invertTransform([0, 0, 0], ori_com)

    # Velocities in global frame
    lin_velocity = data_gps.com_lin_velocity(iteration, sensor_i)
    ang_velocity = data_gps.com_ang_velocity(iteration, sensor_i)

    # Compute velocity in CoM frame
    link_velocity = np.array(pybullet.multiplyTransforms(
        *global2com,
        lin_velocity,
        [0, 0, 0, 1],
    )[0])
    link_angular_velocity = np.array(pybullet.multiplyTransforms(
        *global2com,
        ang_velocity,
        [0, 0, 0, 1],
    )[0])
    urdf2com = pybullet.multiplyTransforms(
        *global2com,
        *urdf2global,
    )
    return (
        link_velocity,
        link_angular_velocity,
        global2com,
        urdf2com,
    )


def compute_buoyancy(link, position, global2com, mass, surface, gravity):
    """Compute buoyancy"""
    return np.array(pybullet.multiplyTransforms(
        *global2com,
        [0, 0, -1000*mass*gravity/link.density*min(
            max(surface-position, 0)/link.height, 1
        )],
        [0, 0, 0, 1],
    )[0]) if mass > 0 else np.zeros(3)


def drag_forces(
        iteration,
        data_gps,
        data_hydrodynamics,
        links,
        masses,
        gravity,
        use_buoyancy,
        surface,
):
    """Drag swimming"""
    positions = np.array(data_gps.array[iteration, :, 2], copy=False)
    sensors = np.argwhere(positions < surface)[:, 0]
    if not sensors.shape[0]:
        return []
    links_map = {link.name: link for link in links}
    links_swimming = [
        links_map[data_gps.names[sensor_i]]
        for sensor_i in sensors
    ]
    for sensor_i, link, position in zip(sensors, links_swimming, positions):
        (
            link_velocity,
            link_angular_velocity,
            global2com,
            urdf2com,
        ) = link_swimming_info(
            data_gps=data_gps,
            iteration=iteration,
            sensor_i=sensor_i,
        )

        # Buoyancy forces
        buoyancy = compute_buoyancy(
            link,
            position,
            global2com,
            masses[link.name],
            surface,
            gravity,
        ) if use_buoyancy else np.zeros(3)

        # Drag forces
        hydro_i = data_hydrodynamics.names.index(link.name)
        coefficients = np.array(link.drag_coefficients)
        data_hydrodynamics.set_force(
            iteration,
            hydro_i,
            (
                np.sign(link_velocity)
                *np.array(pybullet.multiplyTransforms(
                    coefficients[0],
                    [0, 0, 0, 1],
                    *urdf2com,
                )[0])
                *link_velocity**2
                + buoyancy
            )
        )
        data_hydrodynamics.set_torque(
            iteration,
            hydro_i,
            (
                np.sign(link_angular_velocity)
                *np.array(pybullet.multiplyTransforms(
                    coefficients[1],
                    [0, 0, 0, 1],
                    *urdf2com,
                )[0])
                *link_angular_velocity**2
            )
        )
    return links_swimming


def swimming_motion(
        iteration,
        data_hydrodynamics,
        model,
        links,
        links_map,
        link_frame,
        units,
):
    """Swimming motion"""
    for link in links:
        # pybullet.LINK_FRAME applies force in inertial frame, not URDF frame
        sensor_i = data_hydrodynamics.names.index(link.name)
        pybullet.applyExternalForce(
            model,
            links_map[link.name],
            forceObj=(
                np.array(data_hydrodynamics.array[iteration, sensor_i, :3])
                *units.newtons
            ),
            posObj=[0, 0, 0],  # pybullet.getDynamicsInfo(model, link)[3]
            flags=pybullet.LINK_FRAME if link_frame else pybullet.WORLD_FRAME
        )
        pybullet.applyExternalTorque(
            model,
            links_map[link.name],
            torqueObj=(
                np.array(data_hydrodynamics.array[iteration, sensor_i, 3:6])
                *units.torques
            ),
            flags=pybullet.LINK_FRAME if link_frame else pybullet.WORLD_FRAME
        )


def swimming_debug(iteration, data_gps, links):
    """Swimming debug"""
    for link in links:
        sensor_i = data_gps.index(link.name)
        joint = np.array(data_gps.urdf_position(iteration, sensor_i))
        joint_ori = np.array(data_gps.urdf_orientation(iteration, sensor_i))
        # com_ori = np.array(data_gps.com_orientation(iteration, sensor_i))
        ori_joint = np.array(
            pybullet.getMatrixFromQuaternion(joint_ori)
        ).reshape([3, 3])
        # ori_com = np.array(
        #     pybullet.getMatrixFromQuaternion(com_ori)
        # ).reshape([3, 3])
        # ori = np.dot(ori_joint, ori_com)
        axis = 0.05
        offset_x = np.dot(ori_joint, np.array([axis, 0, 0]))
        offset_y = np.dot(ori_joint, np.array([0, axis, 0]))
        offset_z = np.dot(ori_joint, np.array([0, 0, axis]))
        pylog.debug('SPH position: {}'.format(np.array(joint)))
        for i, offset in enumerate([offset_x, offset_y, offset_z]):
            color = np.zeros(3)
            color[i] = 1
            pybullet.addUserDebugLine(
                joint,
                joint + offset,
                lineColorRGB=color,
                lineWidth=5,
                lifeTime=1,
            )
