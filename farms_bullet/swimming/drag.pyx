"""Drag forces"""

include 'types.pxd'

import pybullet
import numpy as np
cimport numpy as np

from farms_data.sensors.data_cy cimport HydrodynamicsArrayCy, LinkSensorArrayCy


cdef void quat_conj(
    DTYPEv1 quat,
    DTYPEv1 out,
) nogil:
    """Quaternion conjugate"""
    cdef unsigned int i
    for i in range(3):
        out[i] = -quat[i]  # x, y, z
    out[3] = quat[3]  # w


cdef void quat_mult(
    DTYPEv1 q0,
    DTYPEv1 q1,
    DTYPEv1 out,
    bint full=1,
) nogil:
    """Hamilton product of two quaternions out = q0*q1"""
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
    """Quaternion rotation

    :param vector: Vector to rotate
    :param quat: Quaternion rotation
    :param quat_c: Returned quaternion conjugate
    :param tmp4: Temporary quaternion
    :param out: Rotated vector

    """
    quat_c[3] = 0
    quat_c[0] = vector[0]
    quat_c[1] = vector[1]
    quat_c[2] = vector[2]
    quat_mult(quat, quat_c, tmp4)
    quat_conj(quat, quat_c)
    quat_mult(tmp4, quat_c, out, full=0)


cdef void link_swimming_info(
    LinkSensorArrayCy data_links,
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

    :param data_links: Links data
    :param iteration: Simulation iteration
    :param sensor_i: Sensor index
    :param urdf2global: URDF to global frame transform
    :param com2global: CoM to global frame transform
    :param global2com: Global to CoM frame transform
    :param urdf2com: URDF to CoM frame transform
    :param link_lin_velocity: Link linear velocity in CoM frame
    :param link_ang_velocity: Link angular velocity in CoM frame
    :param quat_c: Temporary conjugate quaternion
    :param tmp4: Temporary quaternion

    """

    # Orientations
    urdf2global = data_links.urdf_orientation_cy(iteration, sensor_i)
    com2global = data_links.com_orientation_cy(iteration, sensor_i)
    quat_conj(com2global, global2com)

    # Compute velocity in CoM frame
    quat_rot(
        data_links.com_lin_velocity_cy(iteration, sensor_i),
        global2com,
        quat_c,
        tmp4,
        link_lin_velocity,
    )
    quat_rot(
        data_links.com_ang_velocity_cy(iteration, sensor_i),
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

    :param force: Returned force applied to the link in CoM frame
    :param link_velocity: Link linear velocity in CoM frame
    :param coefficients: Drag coefficients
    :param urdf2com: URDF to CoM frame transform
    :param com2urdf: CoM to URDF frame transform
    :param buoyancy: Buoyancy force
    :param quat_c: Temporary conjugate quaternion
    :param tmp4: Temporary quaternion
    :param tmp: Temporary quaternion

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

    :param torque: Returned torque applied to the link in CoM frame
    :param link_ang_velocity: Link angular velocity in CoM frame
    :param coefficients: Drag coefficients
    :param urdf2com: URDF to CoM frame transform
    :param com2urdf: CoM to URDF frame transform
    :param quat_c: Temporary conjugate quaternion
    :param tmp4: Temporary quaternion
    :param tmp: Temporary quaternion

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
    """Compute buoyancy

    :param density: Density of the link
    :param height: Height of the link
    :param position: Z position of the CoM in global frame
    :param global2com: Global to CoM frame transform
    :param mass: Mass of the link
    :param surface: Surface height
    :param gravity: Gravity Z component in global frame
    :param buyoancy: Returned buyoancy forvce in CoM frame
    :param quat_c: Temporary conjugate quaternion
    :param tmp4: Temporary quaternion
    :param tmp: Temporary quaternion

    """
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
        LinkSensorArrayCy data_links,
        unsigned int links_index,
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

    The forces and torques are stored into data_hydrodynamics.array in
    the CoM frame

    :param iteration: Simulation iteration
    :param data_links: Links data
    :param links_index: Link data index
    :param data_hydrodynamics: Hydrodynamics data
    :param hydro_index: Hydrodynamics data index
    :param coefficients: Drag coefficients
    :param z3: Temporary array
    :param z4: Temporary array
    :param surface: Surface height
    :param mass: Link mass
    :param height: Link height
    :param density: Link density
    :param gravity: Gravity value
    :param use_buoyancy: Flag for using buoyancy computation
    """
    cdef unsigned int i
    # cdef unsigned int sensor_i = data_links.names.index(link_name)
    # hydro_i = data_hydrodynamics.names.index(link_name)
    cdef double position = data_links.array[iteration, links_index, 2]
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
        data_links=data_links,
        iteration=iteration,
        sensor_i=links_index,
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
    """Swimming motion

    :param iteration: Simulation iterations
    :param data_hydrodynamics: Hydrodynamics data
    :param hydro_index: Hydrodynamics data index
    :param model: Model identity
    :param link_id: Link identity
    :param frame: Force application frame (LINK_FRAME or WORLD_FRAME)
    :param newtons: Newtons scaling
    :param torques: Torques scaling
    :param pos: Position where to apply the force

    """
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


cpdef swimming_debug(iteration, data_links, links):
    """Swimming debug

    :param iteration: Simulation iterations
    :param data_links: Links data
    :param links: Links options

    """
    for link in links:
        sensor_i = data_links.index(link.name)
        joint = np.array(data_links.urdf_position(iteration, sensor_i))
        joint_ori = np.array(data_links.urdf_orientation(iteration, sensor_i))
        # com_ori = np.array(data_links.com_orientation(iteration, sensor_i))
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
    int model,
    int link_id,
    HydrodynamicsArrayCy data_hydrodynamics,
    unsigned int hydro_index,
    object hydrodynamics_plot,
    bint new_active,
    double meters,
    double scale=1,
):
    """Draw hydrodynamics forces

    :param iteration: Simulation iteration
    :param model: Model identity
    :param link_id: Link identity
    :param data_hydrodynamics: Hydrodynamics data
    :param hydro_index: Hydrodynamics data index
    :param hydrodynamics_plot: Hydrodynamcis plotting objects
    :param new_active: Bool to decalre if recently active
    :param meters: Meters scaling
    :param scale: Plot scaling factor

    """
    cdef bint old_active = hydrodynamics_plot[hydro_index][0]
    cdef DTYPEv1 force = data_hydrodynamics.array[iteration, hydro_index, :3]
    if new_active:
        hydrodynamics_plot[hydro_index][0] = True
        hydrodynamics_plot[hydro_index][1] = pybullet.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=scale*np.array(force),
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


cdef class SwimmingHandler:
    """Swimming handler"""

    cdef object animat
    cdef object links
    cdef object hydro
    cdef int model
    cdef int frame
    cdef unsigned int n_links
    cdef bint drag
    cdef bint sph
    cdef bint buoyancy
    cdef bint show_hydrodynamics
    cdef double water_surface
    cdef double meters
    cdef double newtons
    cdef double torques
    cdef double hydrodynamics_scale
    cdef int[:] links_ids
    cdef int[:] links_swimming
    cdef unsigned int[:] links_indices
    cdef unsigned int[:] hydro_indices
    cdef DTYPEv1 masses
    cdef DTYPEv1 heights
    cdef DTYPEv1 densities
    cdef DTYPEv2 z3
    cdef DTYPEv2 z4
    cdef DTYPEv3 links_coefficients

    def __init__(self, animat):
        super(SwimmingHandler, self).__init__()
        self.animat = animat
        self.links = animat.data.sensors.links
        self.hydro = animat.data.sensors.hydrodynamics
        self.model = animat.identity()
        physics_options = animat.options.physics
        self.drag = physics_options.drag
        self.sph = physics_options.sph
        self.buoyancy = physics_options.buoyancy
        self.show_hydrodynamics = animat.options.show_hydrodynamics
        self.water_surface = physics_options.water_surface
        self.frame = pybullet.LINK_FRAME  # pybullet.WORLD_FRAME
        self.meters = animat.units.meters
        self.newtons = animat.units.newtons
        self.torques = animat.units.torques
        self.hydrodynamics_scale = 1
        self.z3 = np.zeros([6, 3])
        self.z4 = np.zeros([7, 4])
        links = [
            link
            for link in self.animat.options.morphology.links
            if link.swimming
        ]
        self.n_links = len(links)
        self.masses = np.array([self.animat.masses[link.name] for link in links])
        aabb = [
            pybullet.getAABB(
                bodyUniqueId=animat.identity(),
                linkIndex=self.animat.links_map[link.name],
            )
            for link in links
        ]
        self.heights = np.array([
            0.5*(_aabb[1][2] -_aabb[0][2])
            for _aabb in aabb
        ])
        self.densities = np.array([link.density for link in links])
        self.hydro_indices = np.array([
            self.hydro.names.index(link.name)
            for link in links
        ], dtype=np.uintc)
        self.links_indices = np.array([
            self.links.names.index(link.name)
            for link in links
        ], dtype=np.uintc)
        self.links_coefficients = np.array([
            np.array(link.drag_coefficients)
            for link in links
        ])
        self.links_ids = np.array([
            self.animat.links_map[link.name]
            for link in links
        ], dtype=np.intc)
        if self.sph:
            self.water_surface = 1e8

    cpdef step(self, unsigned int iteration):
        """Swimming step"""
        cdef unsigned int i
        cdef bint apply_force = 1
        if self.drag or self.sph:
            hydrodynamics_plot = self.animat.hydrodynamics_plot
            for i in range(self.n_links):
                if self.drag:
                    apply_force = drag_forces(
                        iteration=iteration,
                        data_links=self.links,
                        links_index=self.links_indices[i],
                        data_hydrodynamics=self.hydro,
                        hydro_index=self.hydro_indices[i],
                        coefficients=self.links_coefficients[i],
                        z3=self.z3,
                        z4=self.z4,
                        surface=self.water_surface,
                        mass=self.masses[i],
                        height=self.heights[i],
                        density=self.densities[i],
                        gravity=-9.81,
                        use_buoyancy=self.buoyancy,
                    )
                if apply_force:
                    swimming_motion(
                        iteration=iteration,
                        data_hydrodynamics=self.hydro,
                        hydro_index=self.hydro_indices[i],
                        model=self.model,
                        link_id=self.links_ids[i],
                        frame=self.frame,
                        newtons=self.newtons,
                        torques=self.torques,
                    )
                    if False:
                        swimming_debug(
                            iteration=iteration,
                            data_links=self.links,
                            link=link,
                        )
                if self.show_hydrodynamics:
                    draw_hydrodynamics(
                        iteration=iteration,
                        model=self.model,
                        link_id=self.links_ids[i],
                        data_hydrodynamics=self.hydro,
                        hydro_index=self.hydro_indices[i],
                        hydrodynamics_plot=hydrodynamics_plot,
                        new_active=apply_force,
                        meters=self.meters,
                        scale=self.hydrodynamics_scale,
                    )

    cpdef set_hydrodynamics_scale(self, double value):
        """Set hydrodynamics scale"""
        self.hydrodynamics_scale = value
