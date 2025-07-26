"""Drag forces"""

include 'types.pxd'

import numpy as np
cimport numpy as np

from farms_core.sensors.data_cy cimport LinkSensorArrayCy, XfrcArrayCy
from farms_core.utils.transform cimport quat_conj, quat_mult, quat_rot


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
        double time,
        unsigned int iteration,
        LinkSensorArrayCy data_links,
        unsigned int links_index,
        XfrcArrayCy data_xfrc,
        unsigned int xfrc_index,
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

    The forces and torques are stored into data_xfrc.array in
    the CoM frame

    :param time: Simulation time
    :param iteration: Simulation iteration
    :param data_links: Links data
    :param links_index: Link data index
    :param data_xfrc: Xfrc data
    :param xfrc_index: Xfrc data index
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
    cdef double surface = water.surface(time, pos_x, pos_y)
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
        vector=water.velocity(time, pos_x, pos_y, pos_z),
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
        viscosity=water.viscosity(time, pos_x, pos_y, pos_z),
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
        data_xfrc.array[iteration, xfrc_index, i] = force[i]
        data_xfrc.array[iteration, xfrc_index, i+3] = torque[i]
    return 1


cdef class WaterProperties:
    """Water properties"""

    def __init__(self):
        super(WaterProperties, self).__init__()

    cdef double surface(self, double t, double x, double y):  # nogil
        """Surface"""
        return 0

    cdef double density(self, double t, double x, double y, double z):  # nogil
        """Density"""
        return 1000

    cdef DTYPEv1 velocity(self, double t, double x, double y, double z):  # nogil
        """Velocity in global frame"""
        return np.array([0, 0, 0])

    cdef double viscosity(self, double t, double x, double y, double z):  # nogil
        """Viscosity"""
        return 1.0


cdef class WaterPropertiesConstant(WaterProperties):
    """Water properties"""

    cdef double _surface
    cdef double _density
    cdef double _viscosity
    cdef DTYPEv1 _velocity

    def __init__(self, surface, density, velocity, viscosity):
        super(WaterPropertiesConstant, self).__init__()
        self._surface = surface
        self._density = density
        self._velocity = velocity
        self._viscosity = viscosity

    cdef double surface(self, double t, double x, double y):  # nogil
        """Surface"""
        return self._surface

    cdef double density(self, double t, double x, double y, double z):  # nogil
        """Density"""
        return self._density

    cdef DTYPEv1 velocity(self, double t, double x, double y, double z):  # nogil
        """Velocity in global frame"""
        return self._velocity

    cdef double viscosity(self, double t, double x, double y, double z):  # nogil
        """Viscosity"""
        return self._viscosity

    cpdef void set_velocity(self, double t, double vx, double vy, double vz):
        """Set velocity"""
        self._velocity[0] = vx
        self._velocity[1] = vy
        self._velocity[2] = vz


cdef class WaterPropertiesCallback(WaterProperties):
    """Water properties"""

    cdef object _surface
    cdef object _density
    cdef object _viscosity
    cdef object _velocity

    def __init__(self, surface, density, velocity, viscosity):
        super(WaterPropertiesCallback, self).__init__()
        self._surface = surface
        self._density = density
        self._velocity = velocity
        self._viscosity = viscosity

    cdef double surface(self, double t, double x, double y):  # nogil
        """Surface"""
        return self._surface(t, x, y)

    cdef double density(self, double t, double x, double y, double z):  # nogil
        """Density"""
        return self._density(t, x, y, z)

    cdef DTYPEv1 velocity(self, double t, double x, double y, double z):  # nogil
        """Velocity in global frame"""
        return self._velocity(t, x, y, z)

    cdef double viscosity(self, double t, double x, double y, double z):  # nogil
        """Viscosity"""
        return self._viscosity(t, x, y, z)


cdef class SwimmingHandler:
    """Swimming handler"""

    cdef object links
    cdef object xfrc
    cdef object animat_options
    cdef unsigned int n_links
    cdef bint drag
    cdef bint sph
    cdef bint buoyancy
    cdef WaterProperties water
    cdef double meters
    cdef double newtons
    cdef double torques
    cdef int[:] links_swimming
    cdef unsigned int[:] links_indices
    cdef unsigned int[:] xfrc_indices
    cdef DTYPEv1 masses
    cdef DTYPEv1 heights
    cdef DTYPEv1 densities
    cdef DTYPEv2 z3
    cdef DTYPEv2 z4
    cdef DTYPEv3 links_coefficients

    def __init__(self, data, animat_options, arena_options, units, physics, water=None):
        super(SwimmingHandler, self).__init__()
        self.animat_options = animat_options
        self.links = data.sensors.links
        self.xfrc = data.sensors.xfrc
        water_options = arena_options.water
        self.drag = bool(water_options.drag)
        self.sph = getattr(water_options, 'sph', False)
        self.buoyancy = bool(water_options.buoyancy)
        self.meters = float(units.meters)
        self.newtons = float(units.newtons)
        self.torques = float(units.torques)
        self.water = WaterPropertiesConstant(
            surface=float(water_options.height),
            density=float(water_options.density),
            velocity=np.array(water_options.velocity, dtype=float),
            viscosity=float(water_options.viscosity),
        ) if water is None else water
        self.z3 = np.zeros([7, 3])
        self.z4 = np.zeros([7, 4])
        links = [
            link
            for link in self.animat_options.morphology.links
            if link.swimming
        ]
        self.n_links = len(links)
        links_row = physics.named.model.body_mass.axes.row
        self.masses = np.array([
            physics.model.body_mass[links_row.convert_key_item(link.name)]
            for link in links
        ], dtype=float)/units.kilograms
        self.heights = np.array([
            [
                0.5*physics.model.geom_rbound[geom_i]
                for geom_i in range(len(physics.model.geom_bodyid))
                if links_row.names[physics.named.model.geom_bodyid[geom_i]]
                == link.name
            ][0]
            for link in links
        ], dtype=float)/self.meters
        self.densities = np.array([link.density for link in links])
        self.xfrc_indices = np.array([
            self.xfrc.names.index(link.name)
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
        if self.sph:
            self.water._surface = 1e8

    cpdef step(self, double time, unsigned int iteration):
        """Swimming step"""
        cdef unsigned int i
        cdef bint apply_force = 1
        if self.drag or self.sph:
            for i in range(self.n_links):
                if self.drag:
                    apply_force = drag_forces(
                        time=time,
                        iteration=iteration,
                        data_links=self.links,
                        links_index=self.links_indices[i],
                        data_xfrc=self.xfrc,
                        xfrc_index=self.xfrc_indices[i],
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

    cpdef set_frame(self, int frame):
        """Set frame"""
        self.frame = frame

    cpdef void set_water_velocity(self, DTYPEv1 velocity):
        """Set water velocity"""
        self.water.set_velocity(vx=velocity[0], vy=velocity[1], vz=velocity[2])
