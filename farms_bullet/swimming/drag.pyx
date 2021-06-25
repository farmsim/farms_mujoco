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
    DTYPEv1 global2urdf,
    DTYPEv1 com2urdf,
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
    :param global2urdf: Global to URDF frame transform
    :param urdf2com: Returned URDF to CoM frame transform
    :param link_lin_velocity: Link linear velocity in URDF frame
    :param link_ang_velocity: Link angular velocity in URDF frame
    :param quat_c: Temporary conjugate quaternion
    :param tmp4: Temporary quaternion

    """

    # Orientations
    urdf2global = data_links.urdf_orientation_cy(iteration, sensor_i)
    com2global = data_links.com_orientation_cy(iteration, sensor_i)
    quat_conj(urdf2global, global2urdf)
    quat_mult(global2urdf, com2global, com2urdf)
    quat_conj(com2urdf, urdf2com)

    # Compute velocity in CoM frame
    quat_rot(
        data_links.com_lin_velocity_cy(iteration, sensor_i),
        global2urdf,
        quat_c,
        tmp4,
        link_lin_velocity,
    )
    quat_rot(
        data_links.com_ang_velocity_cy(iteration, sensor_i),
        global2urdf,
        quat_c,
        tmp4,
        link_ang_velocity,
    )


cdef void compute_force(
    DTYPEv1 force,
    DTYPEv1 link_velocity,
    DTYPEv1 coefficients,
    DTYPEv1 buoyancy,
    double viscosity,
) nogil:
    """Compute force and torque

    :param force: Returned force applied to the link in URDF frame
    :param link_velocity: Link linear velocity in URDF frame
    :param coefficients: Drag coefficients
    :param buoyancy: Buoyancy force
    :param viscosity: Fluid viscosity

    """
    cdef unsigned int i
    for i in range(3):
        force[i] = link_velocity[i]*link_velocity[i]
        if link_velocity[i] < 0:
            force[i] *= -1
        force[i] *= viscosity*coefficients[i]
        force[i] += buoyancy[i]


cdef void compute_torque(
    DTYPEv1 torque,
    DTYPEv1 link_ang_velocity,
    DTYPEv1 coefficients,
) nogil:
    """Compute force and torque

    :param torque: Returned torque applied to the link in CoM frame
    :param link_ang_velocity: Link angular velocity in CoM frame
    :param coefficients: Drag coefficients

    """
    cdef unsigned int i
    for i in range(3):
        torque[i] = link_ang_velocity[i]*link_ang_velocity[i]
        if link_ang_velocity[i] < 0:
            torque[i] *= -1
        torque[i] *= coefficients[i]


cdef void compute_buoyancy(
    double density,
    double height,
    double position,
    DTYPEv1 global2urdf,
    double mass,
    double surface,
    double gravity,
    DTYPEv1 buoyancy,
    DTYPEv1 quat_c,
    DTYPEv1 tmp4,
    DTYPEv1 tmp,
) nogil:
    """Compute buoyancy

    :param density: Density of the link
    :param height: Height of the link
    :param position: Z position of the CoM in global frame
    :param global2urdf: Global to URDF frame transform
    :param mass: Mass of the link
    :param surface: Surface height
    :param gravity: Gravity Z component in global frame
    :param buoyancy: Returned buoyancy force in URDF frame
    :param quat_c: Temporary conjugate quaternion
    :param tmp4: Temporary quaternion
    :param tmp: Temporary quaternion

    """
    if mass > 0 and position < surface:
        tmp[0] = 0
        tmp[1] = 0
        tmp[2] = -1000*mass*gravity/density*min(
            max(surface-position, 0)/height,
            1,
        )
        quat_rot(tmp, global2urdf, quat_c, tmp4, buoyancy)
    else:
        for i in range(3):
            buoyancy[i] = 0


cpdef bint drag_forces(
        unsigned int iteration,
        LinkSensorArrayCy data_links,
        unsigned int links_index,
        HydrodynamicsArrayCy data_hydrodynamics,
        unsigned int hydro_index,
        DTYPEv2 coefficients,
        DTYPEv2 z3,
        DTYPEv2 z4,
        WaterProperties water,
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
    :param water: Water properties
    :param mass: Link mass
    :param height: Link height
    :param density: Link density
    :param gravity: Gravity value
    :param use_buoyancy: Flag for using buoyancy computation
    """
    cdef unsigned int i
    cdef double pos_x = data_links.array[iteration, links_index, 0]
    cdef double pos_y = data_links.array[iteration, links_index, 1]
    cdef double pos_z = data_links.array[iteration, links_index, 2]
    cdef double surface = water.surface(pos_x, pos_y)
    if pos_z > surface:
        return 0
    cdef DTYPEv1 force=z3[0], torque=z3[1], buoyancy=z3[2], tmp=z3[3]
    cdef DTYPEv1 link_lin_velocity=z3[4], link_ang_velocity=z3[5]
    cdef DTYPEv1 fluid_velocity_urdf=z3[6]
    cdef DTYPEv1 urdf2global=z4[0], com2global=z4[1]
    cdef DTYPEv1 global2urdf=z4[2], urdf2com=z4[3], com2urdf=z4[4]
    cdef DTYPEv1 quat_c=z4[5], tmp4=z4[6]

    # Swimming information
    link_swimming_info(
        data_links=data_links,
        iteration=iteration,
        sensor_i=links_index,
        urdf2global=urdf2global,
        com2global=com2global,
        global2urdf=global2urdf,
        com2urdf=com2urdf,
        urdf2com=urdf2com,
        link_lin_velocity=link_lin_velocity,
        link_ang_velocity=link_ang_velocity,
        quat_c=quat_c,
        tmp4=tmp4,
    )

    # Buoyancy forces
    if use_buoyancy:
        compute_buoyancy(
            density=density,
            height=height,
            position=pos_z,
            global2urdf=global2urdf,
            mass=mass,
            surface=surface,
            gravity=gravity,
            buoyancy=buoyancy,
            quat_c=quat_c,
            tmp4=tmp4,
            tmp=tmp,
        )

    # Add fluid velocity
    quat_rot(
        vector=water.velocity(pos_x, pos_y, pos_z),
        quat=global2urdf,
        quat_c=quat_c,
        tmp4=tmp4,
        out=fluid_velocity_urdf,
    )
    link_lin_velocity[0] -= fluid_velocity_urdf[0]
    link_lin_velocity[1] -= fluid_velocity_urdf[1]
    link_lin_velocity[2] -= fluid_velocity_urdf[2]

    # Drag forces in URDF frame
    compute_force(
        force=force,
        link_velocity=link_lin_velocity,
        coefficients=coefficients[0],
        buoyancy=buoyancy,
        viscosity=water.viscosity(pos_x, pos_y, pos_z),
    )
    compute_torque(
        torque=torque,
        link_ang_velocity=link_ang_velocity,
        coefficients=coefficients[1],
    )

    # Drag forces in inertial frame
    quat_rot(force, urdf2com, quat_c, tmp4, force)
    quat_rot(torque, urdf2com, quat_c, tmp4, torque)

    # Store data
    for i in range(3):
        data_hydrodynamics.array[iteration, hydro_index, i] = force[i]
        data_hydrodynamics.array[iteration, hydro_index, i+3] = torque[i]
    return 1


cpdef void swimming_apply_forces(
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
    cdef unsigned int i  # , sensor_i, flags
    cdef np.ndarray hydro_force=np.zeros(3), hydro_torque=np.zeros(3)
    cdef DTYPEv1 hydro = data_hydrodynamics.array[iteration, hydro_index]
    for i in range(3):
        hydro_force[i] = hydro[i]*newtons
        hydro_torque[i] = hydro[i+3]*torques
    # pybullet.LINK_FRAME applies force in inertial frame, not URDF frame
    pybullet.applyExternalForce(
        model,
        link_id,
        forceObj=hydro_force.tolist(),
        posObj=pos.tolist(),
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


cdef class WaterProperties:
    """Water properties"""

    cdef double _surface
    cdef double _density
    cdef double _viscosity
    cdef DTYPEv1 _velocity

    def __init__(self, surface, density, velocity, viscosity):
        super(WaterProperties, self).__init__()
        self._surface = surface
        self._density = density
        self._velocity = velocity
        self._viscosity = viscosity

    cdef double surface(self, double x, double y) nogil:
        """Surface"""
        return self._surface

    cdef double density(self, double x, double y, double z) nogil:
        """Density"""
        return self._density

    cdef DTYPEv1 velocity(self, double x, double y, double z) nogil:
        """Velocity in global frame"""
        return self._velocity

    cdef double viscosity(self, double x, double y, double z) nogil:
        """Viscosity"""
        return self._viscosity


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
    cdef WaterProperties water
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
        self.meters = animat.units.meters
        self.newtons = animat.units.newtons
        self.torques = animat.units.torques
        self.water = WaterProperties(
            surface=physics_options.water_surface,
            density=physics_options.water_density,
            velocity=np.array(physics_options.water_velocity),
            viscosity=physics_options.viscosity,
        )
        # pybullet.LINK_FRAME applies force in inertial frame, not URDF frame
        self.frame = pybullet.LINK_FRAME  # pybullet.WORLD_FRAME
        self.hydrodynamics_scale = 1*self.meters
        self.z3 = np.zeros([7, 3])
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
        ])/self.meters
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
            self.water._surface = 1e8

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
                        water=self.water,
                        mass=self.masses[i],
                        height=self.heights[i],
                        density=self.densities[i],
                        gravity=-9.81,
                        use_buoyancy=self.buoyancy,
                    )
                if apply_force:
                    swimming_apply_forces(
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
        self.hydrodynamics_scale = value*self.meters

    cpdef set_frame(self, int frame):
        """Set frame"""
        self.frame = frame
