"""Drag forces"""

include 'types.pxd'

import pybullet
import numpy as np
cimport numpy as np

from farms_data.sensors.data_cy cimport HydrodynamicsArrayCy, GpsArrayCy


# cdef link_swimming_info(GpsArrayCy data_gps, unsigned int iteration, int sensor_i):
#     """Link swimming information

#     Times:
#     - 10.369 [s]
#     - 9.403 [s]
#     - 8.972 [s]
#     - 7.815 [s]
#     - 7.204 [s]
#     - 4.507 [s]
#     - 4.304 [s]
#     """

#     # Declarations
#     zeros = [0, 0, 0]
#     quat_unit = [0, 0, 0, 1]

#     # Orientations
#     ori_urdf = np.array(
#         data_gps.urdf_orientation_cy(iteration, sensor_i),
#         copy=False,
#     ).tolist()
#     ori_com = np.array(
#         data_gps.com_orientation_cy(iteration, sensor_i),
#         copy=False,
#     ).tolist()
#     global2com = pybullet.invertTransform(zeros, ori_com)
#     # urdf2global = (zeros, ori_urdf)

#     # Velocities in global frame
#     lin_velocity = np.array(
#         data_gps.com_lin_velocity_cy(iteration, sensor_i),
#         copy=False,
#     ).tolist()
#     ang_velocity = np.array(
#         data_gps.com_ang_velocity_cy(iteration, sensor_i),
#         copy=False,
#     ).tolist()

#     # Compute velocity in CoM frame
#     link_velocity = np.array(pybullet.multiplyTransforms(
#         *global2com,
#         lin_velocity,
#         quat_unit,
#     )[0])
#     link_angular_velocity = np.array(pybullet.multiplyTransforms(
#         *global2com,
#         ang_velocity,
#         quat_unit,
#     )[0])
#     urdf2com = pybullet.multiplyTransforms(
#         *global2com,
#         zeros,  # *urdf2global,
#         ori_urdf,  # *urdf2global,
#     )
#     return (
#         link_velocity,
#         link_angular_velocity,
#         global2com,
#         urdf2com,
#     )


# cdef void compute_force(np.ndarray force, np.ndarray link_velocity, np.ndarray coefficients, urdf2com, np.ndarray buoyancy):
#     """Compute force and torque

#     Times:
#     - 3.533 [s]
#     - 3.243 [s]
#     """
#     force[:] = (
#         np.sign(link_velocity)
#         *np.array(pybullet.multiplyTransforms(
#             coefficients[0],
#             [0, 0, 0, 1],
#             *urdf2com,
#         )[0])
#         *link_velocity**2
#         + buoyancy
#     )


# cdef void compute_torque(np.ndarray torque, np.ndarray link_angular_velocity, np.ndarray coefficients, urdf2com):
#     """Compute force and torque

#     Times:
#     - 3.533 [s]
#     - 3.243 [s]
#     """
#     torque[:] = (
#         np.sign(link_angular_velocity)
#         *np.array(pybullet.multiplyTransforms(
#             coefficients[1],
#             [0, 0, 0, 1],
#             *urdf2com,
#         )[0])
#         *link_angular_velocity**2
#     )


# cpdef list drag_forces(
#         unsigned int iteration,
#         GpsArrayCy data_gps,
#         HydrodynamicsArrayCy data_hydrodynamics,
#         links,
#         masses,
#         double gravity,
#         use_buoyancy,
#         double surface,
# ):
#     """Drag swimming

#     Times:
#     - 14.967 [s]
#     - 13.877 [s]
#     -  [s]
#     -  [s]
#     -  [s]
#     -  [s]
#     """
#     cdef unsigned int i, hydro_i
#     cdef np.ndarray force = np.zeros(3)
#     cdef np.ndarray torque = np.zeros(3)
#     cdef np.ndarray coefficients, buoyancy
#     cdef np.ndarray positions = np.array(data_gps.array[iteration, :, 2], copy=False)
#     cdef np.ndarray sensors = np.argwhere(positions < surface)[:, 0]
#     if not sensors.shape[0]:
#         return []
#     links_map = {link.name: link for link in links}
#     links_swimming = [
#         links_map[data_gps.names[sensor_i]]
#         for sensor_i in sensors
#         if data_gps.names[sensor_i] in links_map
#     ]
#     for sensor_i, link, position in zip(sensors, links_swimming, positions):
#         (
#             link_velocity,
#             link_angular_velocity,
#             global2com,
#             urdf2com,
#         ) = link_swimming_info(
#             data_gps=data_gps,
#             iteration=iteration,
#             # sensor_i=sensor_i,
#             sensor_i=data_gps.names.index(link.name),
#         )

#         # Buoyancy forces
#         buoyancy = compute_buoyancy(
#             link,
#             position,
#             global2com,
#             masses[link.name],
#             surface,
#             gravity,
#         ) if use_buoyancy else np.zeros(3)

#         # Drag forces in inertial frame
#         coefficients = np.array(link.drag_coefficients)
#         compute_force(
#             force,
#             link_velocity=link_velocity,
#             coefficients=coefficients,
#             urdf2com=urdf2com,
#             buoyancy=buoyancy,
#         )
#         compute_torque(
#             torque,
#             link_angular_velocity=link_angular_velocity,
#             coefficients=coefficients,
#             urdf2com=urdf2com,
#         )
#         hydro_i = data_hydrodynamics.names.index(link.name)
#         for i in range(3):
#             data_hydrodynamics.array[iteration, hydro_i, i] = force[i]
#             data_hydrodynamics.array[iteration, hydro_i, i+3] = torque[i]
#     return links_swimming


cdef void transpose(
    DTYPEv2 matrix,
    DTYPEv2 out,
) nogil:
    """Quaternion to matrix"""
    cdef unsigned int i, j
    for i in range(3):
        for j in range(3):
            out[j, i] = matrix[i, j]


cdef void matrix_dot_vector(
    DTYPEv2 matrix,
    DTYPEv1 vector,
    DTYPEv1 out,
) nogil:
    """Matrix-vector multiplication"""
    cdef unsigned int i, j
    for i in range(3):
        out[i] = 0
        for j in range(3):
            out[i] += matrix[i, j]*vector[j]


cdef void matrix_dot_matrix(
    DTYPEv2 matrix,
    DTYPEv2 matrix2,
    DTYPEv2 out,
) nogil:
    """Matrix-vector multiplication"""
    cdef unsigned int i, j, k
    for i in range(3):
        for j in range(3):
            out[i, j] = 0
            for k in range(3):
                out[i, j] += matrix[i, k]*matrix2[k, j]


cdef void quat_mult(
    DTYPEv1 q0,
    DTYPEv1 q1,
    DTYPEv1 out,
) nogil:
    """Quaternion multiplication"""
    out[0] = q0[3]*q1[0] + q0[0]*q1[3] + q0[1]*q1[2] - q0[2]*q1[1]  # x
    out[1] = q0[3]*q1[1] - q0[0]*q1[2] + q0[1]*q1[3] + q0[2]*q1[0]  # y
    out[2] = q0[3]*q1[2] + q0[0]*q1[1] - q0[1]*q1[0] + q0[2]*q1[3]  # z
    out[3] = q0[3]*q1[3] - q0[0]*q1[0] - q0[1]*q1[1] - q0[2]*q1[2]  # w


cdef void quat_rot(
    DTYPEv1 vector,
    DTYPEv1 quat,
    DTYPEv1 quat_conj,
    DTYPEv1 tmp4,
    DTYPEv1 out,
) nogil:
    """Quaternion rotation"""
    for i in range(3):
        quat_conj[i] = vector[i]
    quat_conj[3] = 0
    quat_mult(quat, quat_conj, tmp4)
    for i in range(3):
        quat_conj[i] = -quat[i]
    quat_conj[3] = quat[3]
    quat_mult(tmp4, quat_conj, out)


cdef void quat2rot(DTYPEv1 quat, DTYPEv2 matrix) nogil:
    """Quaternion to matrix"""
    cdef double q00 = quat[0]*quat[0]
    cdef double q01 = quat[0]*quat[1]
    cdef double q02 = quat[0]*quat[2]
    cdef double q03 = quat[0]*quat[3]
    cdef double q11 = quat[1]*quat[1]
    cdef double q12 = quat[1]*quat[2]
    cdef double q13 = quat[1]*quat[3]
    cdef double q22 = quat[2]*quat[2]
    cdef double q23 = quat[2]*quat[3]
    matrix[0, 0] = 1 - 2*(q11 + q22)
    matrix[0, 1] = 2*(q01 - q23)
    matrix[0, 2] = 2*(q02 + q13)
    matrix[1, 0] = 2*(q01 + q23)
    matrix[1, 1] = 1 - 2*(q00 + q22)
    matrix[1, 2] = 2*(q12 - q03)
    matrix[2, 0] = 2*(q02 - q13)
    matrix[2, 1] = 2*(q12 + q03)
    matrix[2, 2] = 1 - 2*(q00 + q11)


cdef void link_swimming_info(
    GpsArrayCy data_gps,
    unsigned int iteration,
    int sensor_i,
    DTYPEv2 urdf2global,
    DTYPEv2 com2global,
    DTYPEv2 global2com,
    DTYPEv2 urdf2com,
    DTYPEv1 link_lin_velocity,
    DTYPEv1 link_ang_velocity,
) nogil:
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

    # Orientations
    quat2rot(data_gps.urdf_orientation_cy(iteration, sensor_i), urdf2global)
    quat2rot(data_gps.com_orientation_cy(iteration, sensor_i), com2global)
    transpose(com2global, global2com)

    # Compute velocity in CoM frame
    matrix_dot_vector(
        global2com,
        data_gps.com_lin_velocity_cy(iteration, sensor_i),
        link_lin_velocity,
    )
    matrix_dot_vector(
        global2com,
        data_gps.com_ang_velocity_cy(iteration, sensor_i),
        link_ang_velocity,
    )
    matrix_dot_matrix(
        urdf2global,
        global2com,
        urdf2com,
    )


cdef void compute_force(
    DTYPEv1 force,
    DTYPEv1 link_velocity,
    DTYPEv1 coefficients,
    DTYPEv2 urdf2com,
    DTYPEv2 com2urdf,
    DTYPEv1 buoyancy,
    DTYPEv1 tmp,
) nogil:
    """Compute force and torque

    Times:
    - 3.533 [s]
    - 3.243 [s]
    """
    for i in range(3):
        force[i] = link_velocity[i]*link_velocity[i]
        if link_velocity[i] < 0:
            force[i] *= -1
    matrix_dot_vector(
        com2urdf,
        force,
        tmp,
    )
    for i in range(3):
        tmp[i] *= coefficients[i]
    matrix_dot_vector(
        urdf2com,
        tmp,
        force,
    )
    for i in range(3):
        force[i] += buoyancy[i]


cdef void compute_torque(
    DTYPEv1 torque,
    DTYPEv1 link_ang_velocity,
    DTYPEv1 coefficients,
    DTYPEv2 urdf2com,
    DTYPEv2 com2urdf,
    DTYPEv1 tmp,
) nogil:
    """Compute force and torque

    Times:
    - 3.533 [s]
    - 3.243 [s]
    """
    for i in range(3):
        torque[i] = link_ang_velocity[i]*link_ang_velocity[i]
        if link_ang_velocity[i] < 0:
            torque[i] *= -1
    matrix_dot_vector(
        com2urdf,
        torque,
        tmp,
    )
    for i in range(3):
        tmp[i] *= coefficients[i]
    matrix_dot_vector(
        urdf2com,
        tmp,
        torque,
    )


cdef void compute_buoyancy(
    double density,
    double height,
    double position,
    DTYPEv2 global2com,
    double mass,
    double surface,
    double gravity,
    DTYPEv1 buyoancy,
    DTYPEv1 tmp,
) nogil:
    """Compute buoyancy"""
    if mass > 0:
        tmp[0] = 0
        tmp[1] = 0
        tmp[2] = -1000*mass*gravity/density*min(
            max(surface-position, 0)/height,
            1,
        )
        matrix_dot_vector(
            global2com,
            tmp,
            buyoancy,
        )
    else:
        for i in range(3):
            buyoancy[i] = 0


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
    - 11.086 [s]
    -  [s]
    -  [s]
    -  [s]
    """
    cdef unsigned int i, hydro_i
    cdef double[3] force, torque, buoyancy, tmp
    cdef double[3] link_lin_velocity, link_ang_velocity
    cdef double[3][3] urdf2global, com2global, global2com, urdf2com, com2urdf
    cdef DTYPEv2 coefficients
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
        link_swimming_info(
            data_gps=data_gps,
            iteration=iteration,
            sensor_i=data_gps.names.index(link.name),
            urdf2global=urdf2global,
            com2global=com2global,
            global2com=global2com,
            urdf2com=urdf2com,
            link_lin_velocity=link_lin_velocity,
            link_ang_velocity=link_ang_velocity,
        )

        # Buoyancy forces
        if use_buoyancy:
            compute_buoyancy(
                link.density,
                link.height,
                position,
                global2com,
                masses[link.name],
                surface,
                gravity,
                buoyancy,
                tmp,
            )

        # Drag forces in inertial frame
        coefficients = np.array(link.drag_coefficients)
        transpose(urdf2com, com2urdf)
        compute_force(
            force,
            link_velocity=link_lin_velocity,
            coefficients=coefficients[0],
            urdf2com=urdf2com,
            com2urdf=com2urdf,
            buoyancy=buoyancy,
            tmp=tmp,
        )
        compute_torque(
            torque,
            link_ang_velocity=link_ang_velocity,
            coefficients=coefficients[1],
            urdf2com=urdf2com,
            com2urdf=com2urdf,
            tmp=tmp,
        )

        # Store data
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
    cdef DTYPEv1 hydro
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
