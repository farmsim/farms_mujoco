"""Drag forces"""

import pybullet
import numpy as np
cimport numpy as np

from farms_data.sensors.data_cy cimport HydrodynamicsArrayCy, GpsArrayCy


cdef np.ndarray compute_buoyancy(link, np.ndarray position, global2com, double mass, double surface, double gravity):
    """Compute buoyancy"""
    return np.array(pybullet.multiplyTransforms(
        *global2com,
        [0, 0, -1000*mass*gravity/link.density*min(
            max(surface-position, 0)/link.height, 1
        )],
        [0, 0, 0, 1],
    )[0]) if mass > 0 else np.zeros(3)


cdef void compute_force(np.ndarray force, np.ndarray link_velocity, np.ndarray coefficients, urdf2com, np.ndarray buoyancy):
    """Compute force and torque

    Times:
    - 3.533 [s]
    - 3.243 [s]
    """
    force[:] = (
        np.sign(link_velocity)
        *np.array(pybullet.multiplyTransforms(
            coefficients[0],
            [0, 0, 0, 1],
            *urdf2com,
        )[0])
        *link_velocity**2
        + buoyancy
    )


cdef void compute_torque(np.ndarray torque, np.ndarray link_angular_velocity, np.ndarray coefficients, urdf2com):
    """Compute force and torque

    Times:
    - 3.533 [s]
    - 3.243 [s]
    """
    torque[:] = (
        np.sign(link_angular_velocity)
        *np.array(pybullet.multiplyTransforms(
            coefficients[1],
            [0, 0, 0, 1],
            *urdf2com,
        )[0])
        *link_angular_velocity**2
    )


cdef link_swimming_info(GpsArrayCy data_gps, unsigned int iteration, int sensor_i):
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
        unsigned int iteration,
        GpsArrayCy data_gps,
        HydrodynamicsArrayCy data_hydrodynamics,
        links,
        masses,
        double gravity,
        use_buoyancy,
        double surface,
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
    cdef unsigned int i, hydro_i
    cdef np.ndarray force = np.zeros(3)
    cdef np.ndarray torque = np.zeros(3)
    cdef np.ndarray coefficients, buoyancy
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
        coefficients = np.array(link.drag_coefficients)
        compute_force(
            force,
            link_velocity=link_velocity,
            coefficients=coefficients,
            urdf2com=urdf2com,
            buoyancy=buoyancy,
        )
        compute_torque(
            torque,
            link_angular_velocity=link_angular_velocity,
            coefficients=coefficients,
            urdf2com=urdf2com,
        )
        hydro_i = data_hydrodynamics.names.index(link.name)
        for i in range(3):
            data_hydrodynamics.array[iteration, hydro_i, i] = force[i]
            data_hydrodynamics.array[iteration, hydro_i, i+3] = torque[i]
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
