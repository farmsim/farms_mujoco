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
            out[i, j] = matrix[j, i]


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


cdef void quat_conj(
    DTYPEv1 quat,
    DTYPEv1 out,
) nogil:
    """Quaternion multiplication"""
    cdef unsigned int i
    for i in range(3):
        out[i] = -quat[i]
    out[3] = quat[3]


cdef void quat_mult(
    DTYPEv1 q0,
    DTYPEv1 q1,
    DTYPEv1 out,
    bint full=1,
) nogil:
    """Quaternion multiplication"""
    out[0] = q0[3]*q1[0] + q0[0]*q1[3] + q0[1]*q1[2] - q0[2]*q1[1]  # x
    out[1] = q0[3]*q1[1] - q0[0]*q1[2] + q0[1]*q1[3] + q0[2]*q1[0]  # y
    out[2] = q0[3]*q1[2] + q0[0]*q1[1] - q0[1]*q1[0] + q0[2]*q1[3]  # z
    if full:
        out[3] = q0[3]*q1[3] - q0[0]*q1[0] - q0[1]*q1[1] - q0[2]*q1[2]  # w


cdef void quat_rot(
    DTYPEv1 vector,
    DTYPEv1 quat,
    DTYPEv1 quat_c,
    DTYPEv1 tmp4,
    DTYPEv1 out,
) nogil:
    """Quaternion rotation"""
    quat_c[3] = 0
    quat_c[0] = vector[0]
    quat_c[1] = vector[1]
    quat_c[2] = vector[2]
    quat_mult(quat, quat_c, tmp4)
    quat_conj(quat, quat_c)
    quat_mult(tmp4, quat_c, out, full=0)


cdef void quat2rot(DTYPEv1 quat, DTYPEv2 matrix) nogil:
    """Quaternion to rotation matrix"""
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
    DTYPEv1 urdf2global,
    DTYPEv1 com2global,
    DTYPEv1 global2com,
    DTYPEv1 urdf2com,
    DTYPEv1 link_lin_velocity,
    DTYPEv1 link_ang_velocity,
    DTYPEv1 quat_c,
    DTYPEv1 tmp4,
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
    urdf2global = data_gps.urdf_orientation_cy(iteration, sensor_i)
    com2global = data_gps.com_orientation_cy(iteration, sensor_i)
    quat_conj(com2global, global2com)

    # Compute velocity in CoM frame
    quat_rot(
        data_gps.com_lin_velocity_cy(iteration, sensor_i),
        global2com,
        quat_c,
        tmp4,
        link_lin_velocity,
    )
    quat_rot(
        data_gps.com_ang_velocity_cy(iteration, sensor_i),
        global2com,
        quat_c,
        tmp4,
        link_ang_velocity,
    )
    quat_mult(urdf2global, global2com, urdf2com)


cdef void compute_force(
    DTYPEv1 force,
    DTYPEv1 link_velocity,
    DTYPEv1 coefficients,
    DTYPEv1 urdf2com,
    DTYPEv1 com2urdf,
    DTYPEv1 buoyancy,
    DTYPEv1 quat_c,
    DTYPEv1 tmp4,
    DTYPEv1 tmp,
) nogil:
    """Compute force and torque

    Times:
    - 3.533 [s]
    - 3.243 [s]
    """
    cdef unsigned int i
    for i in range(3):
        force[i] = link_velocity[i]*link_velocity[i]
        if link_velocity[i] < 0:
            force[i] *= -1
    quat_rot(force, com2urdf, quat_c, tmp4, tmp)
    for i in range(3):
        tmp[i] *= coefficients[i]
    quat_rot(tmp, urdf2com, quat_c, tmp4, force)
    for i in range(3):
        force[i] += buoyancy[i]


cdef void compute_torque(
    DTYPEv1 torque,
    DTYPEv1 link_ang_velocity,
    DTYPEv1 coefficients,
    DTYPEv1 urdf2com,
    DTYPEv1 com2urdf,
    DTYPEv1 quat_c,
    DTYPEv1 tmp4,
    DTYPEv1 tmp,
) nogil:
    """Compute force and torque

    Times:
    - 3.533 [s]
    - 3.243 [s]
    """
    cdef unsigned int i
    for i in range(3):
        torque[i] = link_ang_velocity[i]*link_ang_velocity[i]
        if link_ang_velocity[i] < 0:
            torque[i] *= -1
    quat_rot(torque, com2urdf, quat_c, tmp4, tmp)
    for i in range(3):
        tmp[i] *= coefficients[i]
    quat_rot(tmp, urdf2com, quat_c, tmp4, torque)


cdef void compute_buoyancy(
    double density,
    double height,
    double position,
    DTYPEv1 global2com,
    double mass,
    double surface,
    double gravity,
    DTYPEv1 buyoancy,
    DTYPEv1 quat_c,
    DTYPEv1 tmp4,
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
        quat_rot(tmp, global2com, quat_c, tmp4, buyoancy)
    else:
        for i in range(3):
            buyoancy[i] = 0


cpdef bint drag_forces(
        unsigned int iteration,
        GpsArrayCy data_gps,
        unsigned int gps_index,
        HydrodynamicsArrayCy data_hydrodynamics,
        unsigned int hydro_index,
        DTYPEv2 coefficients,
        DTYPEv2 z3,
        DTYPEv2 z4,
        double surface,
        double mass,
        double height,
        double density,
        double gravity,
        bint use_buoyancy,
) nogil:
    """Drag swimming

    Times:
    - 14.967 [s]
    - 13.877 [s]
    - 11.086 [s]
    - 2.171 [s]
    -  [s]
    -  [s]
    """
    cdef unsigned int i
    # cdef unsigned int sensor_i = data_gps.names.index(link_name)
    # hydro_i = data_hydrodynamics.names.index(link_name)
    cdef double position = data_gps.array[iteration, gps_index, 2]
    if position > surface:
        return 0
    # cdef double[6][3] z3
    # cdef double[6][4] z4
    # # z3 = np.zeros([6, 3])
    # # z4 = np.zeros([7, 4])
    cdef DTYPEv1 force=z3[0], torque=z3[1], buoyancy=z3[2], tmp=z3[3]
    cdef DTYPEv1 link_lin_velocity=z3[4], link_ang_velocity=z3[5]
    cdef DTYPEv1 urdf2global=z4[0], com2global=z4[1]
    cdef DTYPEv1 global2com=z4[2], urdf2com=z4[3], com2urdf=z4[4]
    cdef DTYPEv1 quat_c=z4[5], tmp4=z4[6]

    # Swimming information
    link_swimming_info(
        data_gps=data_gps,
        iteration=iteration,
        sensor_i=gps_index,
        urdf2global=urdf2global,
        com2global=com2global,
        global2com=global2com,
        urdf2com=urdf2com,
        link_lin_velocity=link_lin_velocity,
        link_ang_velocity=link_ang_velocity,
        quat_c=quat_c,
        tmp4=tmp4,
    )

    # Buoyancy forces
    if use_buoyancy:
        compute_buoyancy(
            density,
            height,
            position,
            global2com,
            mass,
            surface,
            gravity,
            buoyancy,
            quat_c,
            tmp4,
            tmp,
        )

    # Drag forces in inertial frame
    quat_conj(urdf2com, com2urdf)
    compute_force(
        force,
        link_velocity=link_lin_velocity,
        coefficients=coefficients[0],
        urdf2com=urdf2com,
        com2urdf=com2urdf,
        buoyancy=buoyancy,
        quat_c=quat_c,
        tmp4=tmp4,
        tmp=tmp,
    )
    compute_torque(
        torque,
        link_ang_velocity=link_ang_velocity,
        coefficients=coefficients[1],
        urdf2com=urdf2com,
        com2urdf=com2urdf,
        quat_c=quat_c,
        tmp4=tmp4,
        tmp=tmp,
    )

    # Store data
    for i in range(3):
        data_hydrodynamics.array[iteration, hydro_index, i] = force[i]
        data_hydrodynamics.array[iteration, hydro_index, i+3] = torque[i]
    return 1


cpdef void swimming_motion(
        unsigned int iteration,
        HydrodynamicsArrayCy data_hydrodynamics,
        unsigned int hydro_index,
        int model,
        int link_id,
        int frame=pybullet.LINK_FRAME,
        double newtons=1.0,
        double torques=1.0,
        np.ndarray pos=np.zeros(3),
):
    """Swimming motion"""
    # cdef int link_id
    # cdef str link_name
    cdef unsigned int i  # , sensor_i, flags
    cdef np.ndarray hydro_force=np.zeros(3), hydro_torque=np.zeros(3)
    # cdef np.ndarray hydro
    # cdef double newtons, torques
    # newtons = units.newtons
    # torques = units.torques
    # flags = pybullet.LINK_FRAME if link_frame else pybullet.WORLD_FRAME

    # pybullet.LINK_FRAME applies force in inertial frame, not URDF frame
    # sensor_i = data_hydrodynamics.names.index(link.name)
    # link_id = links_map[link.name]
    cdef DTYPEv1 hydro = data_hydrodynamics.array[iteration, hydro_index]
    for i in range(3):
        hydro_force[i] = hydro[i]*newtons
        hydro_torque[i] = hydro[i+3]*torques
    pybullet.applyExternalForce(
        model,
        link_id,
        forceObj=hydro_force.tolist(),
        posObj=pos.tolist(),  # pybullet.getDynamicsInfo(model, link)[3]
        flags=frame,
    )
    pybullet.applyExternalTorque(
        model,
        link_id,
        torqueObj=hydro_torque.tolist(),
        flags=frame,
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


cdef draw_hydrodynamics(
    unsigned int iteration,
    HydrodynamicsArrayCy data_hydrodynamics,
    int model,
    int link_id,
    unsigned int hydro_index,
    hydrodynamics_plot,
    bint new_active,
    double meters,
):
    """Draw hydrodynamics forces"""
    cdef bint old_active = hydrodynamics_plot[hydro_index][0]
    cdef DTYPEv1 force = data_hydrodynamics.array[iteration, hydro_index, :3]
    if new_active:
        hydrodynamics_plot[hydro_index][0] = True
        hydrodynamics_plot[hydro_index][1] = pybullet.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=1000*np.array(force),
            lineColorRGB=[0, 0, 1],
            lineWidth=7*meters,
            parentObjectUniqueId=model,
            parentLinkIndex=link_id,
            replaceItemUniqueId=hydrodynamics_plot[hydro_index][1],
        )
    elif old_active and not new_active:
        hydrodynamics_plot[hydro_index][0] = False
        hydrodynamics_plot[hydro_index][1] = pybullet.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[0, 0, 0],
            lineColorRGB=[0, 0, 1],
            lineWidth=0,
            parentObjectUniqueId=model,
            parentLinkIndex=0,
            replaceItemUniqueId=hydrodynamics_plot[hydro_index][1],
        )


cpdef void swimming_step(iteration, animat):
    """Swimming step"""
    physics_options = animat.options.physics
    cdef bint show_hydrodynamics = animat.options.show_hydrodynamics
    cdef str link_name
    cdef bint apply_force = 1
    cdef unsigned int gps_index
    cdef unsigned int hydro_index
    cdef bint drag = physics_options.drag
    cdef bint sph = physics_options.sph
    cdef bint buoyancy = physics_options.buoyancy
    cdef double water_surface = physics_options.water_surface
    cdef int frame = pybullet.LINK_FRAME
    cdef int model, link_id
    cdef double meters = animat.units.meters
    cdef double newtons = animat.units.newtons
    cdef double torques = animat.units.torques
    cdef double mass
    cdef double height
    cdef double density
    cdef DTYPEv2 z3 = np.zeros([6, 3])
    cdef DTYPEv2 z4 = np.zeros([7, 4])
    cdef DTYPEv2 coefficients
    if drag or sph:
        if sph:
            water_surface = 1e8
        sensors = animat.data.sensors
        hydro = sensors.hydrodynamics
        gps = sensors.gps
        frame = pybullet.LINK_FRAME  # pybullet.WORLD_FRAME
        newtons = animat.units.newtons
        torques = animat.units.torques
        model = animat.identity()
        hydrodynamics_plot = animat.hydrodynamics_plot
        for link in animat.options.morphology.links:
            if link.swimming:
                link_name = link.name
                hydro_index = hydro.names.index(link_name)
                if drag:
                    gps_index = gps.names.index(link_name)
                    coefficients = np.array(link.drag_coefficients)
                    mass = animat.masses[link_name]
                    height = link.height
                    density = link.density
                    apply_force = drag_forces(
                        iteration=iteration,
                        data_gps=gps,
                        gps_index=gps_index,
                        data_hydrodynamics=hydro,
                        hydro_index=hydro_index,
                        coefficients=coefficients,
                        z3=z3,
                        z4=z4,
                        surface=water_surface,
                        mass=mass,
                        height=height,
                        density=density,
                        gravity=-9.81,
                        use_buoyancy=buoyancy,
                    )
                if apply_force:
                    link_id = animat.links_map[link_name]
                    swimming_motion(
                        iteration=iteration,
                        data_hydrodynamics=hydro,
                        hydro_index=hydro_index,
                        model=model,
                        link_id=link_id,
                        frame=frame,
                        newtons=newtons,
                        torques=torques,
                    )
                    if False:
                        swimming_debug(
                            iteration=iteration,
                            data_gps=animat.data.sensors.gps,
                            link=link,
                        )
                if show_hydrodynamics:
                    draw_hydrodynamics(
                        iteration=iteration,
                        data_hydrodynamics=hydro,
                        model=model,
                        link_id=link_id,
                        hydro_index=hydro_index,
                        hydrodynamics_plot=hydrodynamics_plot,
                        new_active=apply_force,
                        meters=meters,
                    )
