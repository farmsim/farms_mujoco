"""Drag forces"""

import pybullet
import numpy as np
cimport numpy as np

from farms_data.sensors.data_cy cimport HydrodynamicsArrayCy, GpsArrayCy
import farms_pylog as pylog


cdef np.ndarray compute_buoyancy(link, position, global2com, mass, surface, gravity):
    """Compute buoyancy"""
    return np.array(pybullet.multiplyTransforms(
        *global2com,
        [0, 0, -1000*mass*gravity/link.density*min(
            max(surface-position, 0)/link.height, 1
        )],
        [0, 0, 0, 1],
    )[0]) if mass > 0 else np.zeros(3)


cdef compute_force_torque(link_velocity, link_angular_velocity, coefficients, urdf2com, buoyancy):
    """Compute force and torque

    Times:
    - 3.533 [s]
    - 3.243 [s]
    """
    return (
        np.sign(link_velocity)
        *np.array(pybullet.multiplyTransforms(
            coefficients[0],
            [0, 0, 0, 1],
            *urdf2com,
        )[0])
        *link_velocity**2
        + buoyancy,
        np.sign(link_angular_velocity)
        *np.array(pybullet.multiplyTransforms(
            coefficients[1],
            [0, 0, 0, 1],
            *urdf2com,
        )[0])
        *link_angular_velocity**2
    )


cdef link_swimming_info(GpsArrayCy data_gps, iteration, sensor_i):
    """Link swimming information

    Times:
    - 10.369 [s]
    - 9.403 [s]
    - 8.972 [s]
    - 7.815 [s]
    - 7.204 [s]
    - 4.507 [s]
    - 4.304 [s]
    """

    # Declarations
    zeros = [0, 0, 0]
    quat_unit = [0, 0, 0, 1]

    # Orientations
    ori_urdf = np.array(
        data_gps.urdf_orientation_cy(iteration, sensor_i),
        copy=False,
    ).tolist()
    ori_com = np.array(
        data_gps.com_orientation_cy(iteration, sensor_i),
        copy=False,
    ).tolist()
    global2com = pybullet.invertTransform(zeros, ori_com)
    # urdf2global = (zeros, ori_urdf)

    # Velocities in global frame
    lin_velocity = np.array(
        data_gps.com_lin_velocity_cy(iteration, sensor_i),
        copy=False,
    ).tolist()
    ang_velocity = np.array(
        data_gps.com_ang_velocity_cy(iteration, sensor_i),
        copy=False,
    ).tolist()

    # Compute velocity in CoM frame
    link_velocity = np.array(pybullet.multiplyTransforms(
        *global2com,
        lin_velocity,
        quat_unit,
    )[0])
    link_angular_velocity = np.array(pybullet.multiplyTransforms(
        *global2com,
        ang_velocity,
        quat_unit,
    )[0])
    urdf2com = pybullet.multiplyTransforms(
        *global2com,
        zeros,  # *urdf2global,
        ori_urdf,  # *urdf2global,
    )
    return (
        link_velocity,
        link_angular_velocity,
        global2com,
        urdf2com,
    )


cpdef list drag_forces(
        iteration,
        data_gps,
        data_hydrodynamics,
        links,
        masses,
        gravity,
        use_buoyancy,
        surface,
):
    """Drag swimming

    Times:
    - 14.967 [s]
    - 13.877 [s]
    -  [s]
    -  [s]
    -  [s]
    -  [s]
    """
    cdef np.ndarray force, torque, buoyancy
    cdef np.ndarray positions = np.array(data_gps.array[iteration, :, 2], copy=False)
    cdef np.ndarray sensors = np.argwhere(positions < surface)[:, 0]
    if not sensors.shape[0]:
        return []
    links_map = {link.name: link for link in links}
    links_swimming = [
        links_map[data_gps.names[sensor_i]]
        for sensor_i in sensors
        if data_gps.names[sensor_i] in links_map
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
            # sensor_i=sensor_i,
            sensor_i=data_gps.names.index(link.name),
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

        # Drag forces in inertial frame
        force, torque = compute_force_torque(
            link_velocity=link_velocity,
            link_angular_velocity=link_angular_velocity,
            coefficients=np.array(link.drag_coefficients),
            urdf2com=urdf2com,
            buoyancy=buoyancy,
        )
        hydro_i = data_hydrodynamics.names.index(link.name)
        data_hydrodynamics.set_force(iteration, hydro_i, force)
        data_hydrodynamics.set_torque(iteration, hydro_i, torque)
    return links_swimming


cpdef void swimming_motion(
        unsigned int iteration,
        HydrodynamicsArrayCy data_hydrodynamics,
        int model,
        list links,
        dict links_map,
        link_frame,
        units,
        pos=np.zeros(3)
):
    """Swimming motion"""
    cdef int link_id
    cdef str link_name
    cdef unsigned int i, sensor_i, flags
    cdef double[:] hydro
    cdef double hydro_force[3], hydro_torque[3]
    cdef double newtons, torques
    newtons = units.newtons
    torques = units.torques
    flags = pybullet.LINK_FRAME if link_frame else pybullet.WORLD_FRAME
    for link in links:
        # pybullet.LINK_FRAME applies force in inertial frame, not URDF frame
        sensor_i = data_hydrodynamics.names.index(link.name)
        link_id = links_map[link.name]
        hydro = data_hydrodynamics.array[iteration, sensor_i]
        for i in range(3):
            hydro_force[i] = hydro[i]*newtons
            hydro_torque[i] = hydro[i+3]*torques
        pybullet.applyExternalForce(
            model,
            link_id,
            forceObj=np.array(hydro_force),
            posObj=pos,  # pybullet.getDynamicsInfo(model, link)[3]
            flags=flags,
        )
        pybullet.applyExternalTorque(
            model,
            link_id,
            torqueObj=np.array(hydro_torque),
            flags=flags,
        )


cpdef swimming_debug(iteration, data_gps, links):
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
