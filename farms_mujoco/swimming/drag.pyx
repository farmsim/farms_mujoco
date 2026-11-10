"""Drag forces"""

include 'types.pxd'

from libc.math cimport isfinite, sqrt

import numpy as np
cimport numpy as np

from farms_core import pylog
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
):
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

    # Orientations. Copy into the caller buffers: assigning to a
    # memoryview parameter would only rebind the local name.
    cdef unsigned int i
    cdef DTYPEv1 urdf_orientation = data_links.urdf_orientation_cy(
        iteration, sensor_i,
    )
    cdef DTYPEv1 com_orientation = data_links.com_orientation_cy(
        iteration, sensor_i,
    )
    for i in range(4):
        urdf2global[i] = urdf_orientation[i]
        com2global[i] = com_orientation[i]
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


cdef inline double submersion_fraction(
    double position,
    double height,
    double surface,
) nogil:
    """Submerged fraction of a link, linear in depth.

    Returns 0 when the link is fully above the surface, 1 when fully
    below, and 0.5 when its centre is at the waterline.

    """
    cdef double fraction
    if height <= 0:
        return 1.0 if position <= surface else 0.0
    fraction = (surface + height - position) / (2.0 * height)
    if fraction <= 0.0:
        return 0.0
    if fraction >= 1.0:
        return 1.0
    return fraction


cdef inline double implicit_drag_velocity(
    double velocity,
    double a,
    double c,
    double f_other,
) nogil:
    """Solve c*x*|x| + a*x = a*v + f_other for the stable root.

    ``a`` and ``c`` must be positive. This is the per-axis backward-Euler
    solution used by both the linear and angular drag models.

    """
    cdef double b = a * velocity + f_other
    if b >= 0:
        return (-a + sqrt(a * a + 4.0 * c * b)) / (2.0 * c)
    return -((-a + sqrt(a * a - 4.0 * c * b)) / (2.0 * c))


cdef void compute_force(
    DTYPEv1 force,
    DTYPEv1 link_velocity,
    DTYPEv1 coefficients,
    DTYPEv1 buoyancy,
    double viscosity,
    double mass,
    double dt,
    double submerged,
):
    """Compute linear drag and buoyancy force.

    Semi-implicit (backward-Euler) quadratic drag. The FARMS drag
    coefficients are negative, so the positive drag magnitude is
    ``C = -viscosity * coefficient * submerged``. Per axis, solve

        C*x*|x| + (m/dt)*x = (m/dt)*v + f_other

    for the end-of-step velocity ``x`` and return
    ``m*(x - v)/dt``. This is stable for large timesteps and reduces to
    the legacy explicit form when the implicit branch is unavailable.

    :param force: Returned force applied to the link in URDF frame
    :param link_velocity: Link linear velocity in URDF frame
    :param coefficients: Drag coefficients (negative in FARMS configs)
    :param buoyancy: Buoyancy force
    :param viscosity: Fluid viscosity
    :param mass: Link mass
    :param dt: Physics timestep
    :param submerged: Submerged fraction in [0, 1]

    """
    cdef unsigned int i
    cdef double v, f_other, a, c_drag, v_new
    for i in range(3):
        v = link_velocity[i]
        f_other = buoyancy[i]
        c_drag = -viscosity * coefficients[i] * submerged
        if mass > 0 and dt > 0 and c_drag > 0 and isfinite(c_drag):
            a = mass / dt
            v_new = implicit_drag_velocity(v, a, c_drag, f_other)
            force[i] = mass * (v_new - v) / dt
        else:
            # Explicit fallback, identical to the legacy form for
            # negative coefficients.
            force[i] = -c_drag * v * abs(v) + f_other


cdef void compute_torque(
    DTYPEv1 torque,
    DTYPEv1 link_ang_velocity,
    DTYPEv1 coefficients,
    DTYPEv1 inertia,
    double dt,
    double submerged,
    DTYPEv1 torque_other,
):
    """Compute semi-implicit angular drag.

    Per axis, solve the same backward-Euler quadratic equation as the
    linear drag, with the link inertia replacing the mass:

        C*w_new*|w_new| + (I/dt)*w_new = (I/dt)*w + torque_other

    and return ``I*(w_new - w)/dt``. ``torque_other`` is the constant
    moment-arm torque for the step and is included in the solve, not
    added again by the caller.

    :param torque: Returned total torque in URDF frame
    :param link_ang_velocity: Link angular velocity in URDF frame
    :param coefficients: Angular drag coefficients (negative in configs)
    :param inertia: Diagonal body-frame inertia (real units)
    :param dt: Physics timestep
    :param submerged: Submerged fraction in [0, 1]
    :param torque_other: Constant torque applied during the step

    """
    cdef unsigned int i
    cdef double w, c_drag, a, w_new
    for i in range(3):
        w = link_ang_velocity[i]
        c_drag = -coefficients[i] * submerged
        if (
            inertia[i] > 0
            and isfinite(inertia[i])
            and dt > 0
            and c_drag > 0
            and isfinite(c_drag)
        ):
            a = inertia[i] / dt
            w_new = implicit_drag_velocity(w, a, c_drag, torque_other[i])
            torque[i] = inertia[i] * (w_new - w) / dt
        else:
            # Explicit fallback, identical to the legacy form for
            # negative coefficients.
            torque[i] = -c_drag * w * abs(w) + torque_other[i]


cdef void compute_buoyancy(
    double density,
    double water_density,
    double mass,
    double gravity,
    double submerged,
    DTYPEv1 global2urdf,
    DTYPEv1 buoyancy,
    DTYPEv1 quat_c,
    DTYPEv1 tmp4,
    DTYPEv1 force_global,
):
    """Compute buoyancy

    :param density: Density of the link
    :param water_density: Density of the surrounding fluid
    :param mass: Mass of the link
    :param gravity: Gravity Z component in global frame
    :param submerged: Submerged fraction in [0, 1]
    :param global2urdf: Global to URDF frame transform
    :param buoyancy: Returned buoyancy force in URDF frame
    :param quat_c: Temporary conjugate quaternion
    :param tmp4: Temporary quaternion (4D)
    :param force_global: Force in global frame

    """
    cdef unsigned int i
    if mass > 0 and submerged > 0 and density > 0:
        force_global[0] = 0.0
        force_global[1] = 0.0
        force_global[2] = -water_density*mass*gravity/density*submerged
        quat_rot(force_global, global2urdf, quat_c, tmp4, buoyancy)
    else:
        for i in range(3):
            buoyancy[i] = 0.0


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
        double dt=0.0,
        DTYPEv1 buoyancy_center=None,
        DTYPEv1 inertia=None,
):
    """Drag swimming

    The forces and torques are stored into data_xfrc.array in
    the CoM frame.

    A single submerged fraction, based on the link CoM, scales linear
    drag, angular drag, and buoyancy magnitude. The CoB offset only
    moves the line of action and the water-property sampling point.

    :param time: Simulation time
    :param iteration: Simulation iteration
    :param data_links: Links data
    :param links_index: Link data index
    :param data_xfrc: Xfrc data
    :param xfrc_index: Xfrc data index
    :param coefficients: Drag coefficients (negative in FARMS configs)
    :param z3: Temporary array (at least 7 rows)
    :param z4: Temporary array
    :param water: Water properties
    :param mass: Link mass
    :param height: Link height
    :param density: Link density
    :param gravity: Gravity value
    :param use_buoyancy: Flag for using buoyancy computation
    :param buoyancy_center: Offset of CoB from link origin in URDF frame
    :param inertia: Diagonal inertia in URDF frame (real units). If
        None, angular drag falls back to the explicit form.
    """
    cdef unsigned int i
    cdef double pos_x = data_links.array[iteration, links_index, 0]
    cdef double pos_y = data_links.array[iteration, links_index, 1]
    cdef double pos_z = data_links.array[iteration, links_index, 2]
    cdef double surface = water.surface(time, pos_x, pos_y)
    cdef double submerged = submersion_fraction(pos_z, height, surface)
    cdef DTYPEv1 force=z3[0], torque=z3[1], buoyancy=z3[2], tmp=z3[3]
    cdef DTYPEv1 link_lin_velocity=z3[4], link_ang_velocity=z3[5]
    cdef DTYPEv1 fluid_velocity_urdf=z3[6]
    cdef DTYPEv1 urdf2global=z4[0], com2global=z4[1]
    cdef DTYPEv1 global2urdf=z4[2], urdf2com=z4[3], com2urdf=z4[4]
    cdef DTYPEv1 quat_c=z4[5], tmp4=z4[6]
    cdef DTYPEv1 inertia_vec
    cdef double r_x = 0.0, r_y = 0.0, r_z = 0.0
    cdef double cob_z = pos_z
    cdef bint has_cob_offset

    # Fully above water: nothing to apply. physics2data already cleared this
    # row, so returning leaves the link force-free.
    if submerged <= 0:
        return 0

    # Buoyancy center offset (default [0, 0, 0] = link origin)
    if buoyancy_center is not None:
        r_x = buoyancy_center[0]
        r_y = buoyancy_center[1]
        r_z = buoyancy_center[2]
    has_cob_offset = (r_x != 0.0 or r_y != 0.0 or r_z != 0.0)

    # Swimming information. Must run before the CoB rotation so that
    # urdf2global/com2global contain this link's orientation (D5).
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

    # CoB position in the global frame. Used for water-property
    # sampling and (through r) for the moment arm; the submerged
    # fraction above is based on the link CoM.
    if has_cob_offset:
        quat_rot(buoyancy_center, urdf2global, quat_c, tmp4, tmp)
        cob_z = pos_z + tmp[2]

    # If CoB offset, adjust velocity to CoB: v_CoB = v_CoM + ω × r
    if has_cob_offset:
        link_lin_velocity[0] += (
            link_ang_velocity[1] * r_z - link_ang_velocity[2] * r_y
        )
        link_lin_velocity[1] += (
            link_ang_velocity[2] * r_x - link_ang_velocity[0] * r_z
        )
        link_lin_velocity[2] += (
            link_ang_velocity[0] * r_y - link_ang_velocity[1] * r_x
        )

    # Buoyancy force (magnitude from the single submerged fraction)
    if use_buoyancy:
        compute_buoyancy(
            density=density,
            water_density=water.density(time, pos_x, pos_y, cob_z),
            mass=mass,
            gravity=gravity,
            submerged=submerged,
            global2urdf=global2urdf,
            buoyancy=buoyancy,
            quat_c=quat_c,
            tmp4=tmp4,
            force_global=tmp,
        )
    else:
        for i in range(3):
            buoyancy[i] = 0.0

    # Add fluid velocity
    quat_rot(
        vector=water.velocity(time, pos_x, pos_y, cob_z),
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
        viscosity=water.viscosity(time, pos_x, pos_y, cob_z),
        mass=mass,
        dt=dt,
        submerged=submerged,
    )

    # Constant moment-arm torque for the step: r × F_total. `force`
    # already contains buoyancy, so do not add r × buoyancy again.
    # tmp is free after compute_buoyancy.
    if has_cob_offset:
        tmp[0] = r_y * force[2] - r_z * force[1]
        tmp[1] = r_z * force[0] - r_x * force[2]
        tmp[2] = r_x * force[1] - r_y * force[0]
    else:
        tmp[0] = 0.0
        tmp[1] = 0.0
        tmp[2] = 0.0

    # Angular drag in URDF frame
    if inertia is None:
        # No inertia supplied: explicit fallback. fluid_velocity_urdf is
        # free at this point, so reuse it as a zero inertia vector.
        for i in range(3):
            fluid_velocity_urdf[i] = 0.0
        inertia_vec = fluid_velocity_urdf
    else:
        inertia_vec = inertia
    compute_torque(
        torque=torque,
        link_ang_velocity=link_ang_velocity,
        coefficients=coefficients[1],
        inertia=inertia_vec,
        dt=dt,
        submerged=submerged,
        torque_other=tmp,
    )

    # Drag forces in inertial frame
    quat_rot(force, urdf2com, quat_c, tmp4, force)
    quat_rot(torque, urdf2com, quat_c, tmp4, torque)

    # Accumulate into the shared xfrc row. The row is reset during sensor
    # collection, and other extensions may add their own contribution.
    for i in range(3):
        data_xfrc.array[iteration, xfrc_index, i] += force[i]
        data_xfrc.array[iteration, xfrc_index, i+3] += torque[i]
    return 1


cdef class WaterProperties:
    """Water properties"""

    def __init__(self):
        super(WaterProperties, self).__init__()

    cdef double surface(self, double t, double x, double y):
        """Surface"""
        return 0

    cdef double density(self, double t, double x, double y, double z):
        """Density"""
        return 1000

    cdef DTYPEv1 velocity(self, double t, double x, double y, double z):
        """Velocity in global frame"""
        return np.array([0, 0, 0])

    cdef double viscosity(self, double t, double x, double y, double z):
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

    cdef double surface(self, double t, double x, double y):
        """Surface"""
        return self._surface

    cdef double density(self, double t, double x, double y, double z):
        """Density"""
        return self._density

    cdef DTYPEv1 velocity(self, double t, double x, double y, double z):
        """Velocity in global frame"""
        return self._velocity

    cdef double viscosity(self, double t, double x, double y, double z):
        """Viscosity"""
        return self._viscosity

    cpdef void set_velocity(self, double vx, double vy, double vz):
        """Set velocity"""
        self._velocity[0] = vx
        self._velocity[1] = vy
        self._velocity[2] = vz


cdef class WaterPropertiesExtension(WaterProperties):
    """Water properties"""

    cdef object _surface
    cdef object _density
    cdef object _viscosity
    cdef object _velocity

    def __init__(self, surface, density, velocity, viscosity):
        super(WaterPropertiesExtension, self).__init__()
        self._surface = surface
        self._density = density
        self._velocity = velocity
        self._viscosity = viscosity

    cdef double surface(self, double t, double x, double y):
        """Surface"""
        return self._surface(t, x, y)

    cdef double density(self, double t, double x, double y, double z):
        """Density"""
        return self._density(t, x, y, z)

    cdef DTYPEv1 velocity(self, double t, double x, double y, double z):
        """Velocity in global frame"""
        return self._velocity(t, x, y, z)

    cdef double viscosity(self, double t, double x, double y, double z):
        """Viscosity"""
        return self._viscosity(t, x, y, z)


def body_inertia_urdf(model, body_id):
    """Diagonal body-frame inertia of a link, in model units.

    MuJoCo stores the principal inertia diagonal together with the
    inertial-frame orientation (``body_iquat``, order ``w, x, y, z``).
    The body-frame tensor is ``R diag(I_p) R^T``; only its diagonal is
    used because the per-axis drag solve is decoupled.

    """
    inertia = np.asarray(model.body_inertia[body_id], dtype=float)
    quat = np.asarray(model.body_iquat[body_id], dtype=float)
    norm = float(np.dot(quat, quat))
    if norm < 1e-12:
        return inertia.copy()
    w, x, y, z = quat / np.sqrt(norm)
    rotation = np.array([
        [
            1.0 - 2.0*(y*y + z*z),
            2.0*(x*y - w*z),
            2.0*(x*z + w*y),
        ], [
            2.0*(x*y + w*z),
            1.0 - 2.0*(x*x + z*z),
            2.0*(y*z - w*x),
        ], [
            2.0*(x*z - w*y),
            2.0*(y*z + w*x),
            1.0 - 2.0*(x*x + y*y),
        ],
    ])
    return np.einsum('ki,i,ki->k', rotation, inertia, rotation)


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
    cdef double kilograms
    cdef int[:] links_swimming
    cdef unsigned int[:] links_indices
    cdef unsigned int[:] xfrc_indices
    cdef DTYPEv1 masses
    cdef DTYPEv1 heights
    cdef DTYPEv1 densities
    cdef DTYPEv2 inertias
    cdef DTYPEv2 buoyancy_centers
    cdef DTYPEv2 z3
    cdef DTYPEv2 z4
    cdef DTYPEv3 links_coefficients

    def __init__(self, data, animat_options, arena_options, units, physics, water=None, prefix=''):
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
        self.kilograms = float(units.kilograms)
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
            if link.fluid_interaction
        ]
        self.n_links = len(links)
        links_row = physics.named.model.body_mass.axes.row
        body_ids = [
            links_row.convert_key_item(prefix + link.name)
            for link in links
        ]
        self.masses = np.array([
            physics.model.body_mass[body_id]
            for body_id in body_ids
        ], dtype=float)/self.kilograms
        self.inertias = np.array([
            body_inertia_urdf(physics.model, body_id)
            for body_id in body_ids
        ], dtype=float)/(self.kilograms*self.meters**2)
        link_heights = []
        for link in links:
            matching_heights = [
                0.75*physics.model.geom_rbound[geom_i]
                for geom_i in range(len(physics.model.geom_bodyid))
                if links_row.names[physics.named.model.geom_bodyid[geom_i]]
                == prefix+link.name
            ]
            if not matching_heights:
                raise ValueError(
                    f'No geom found for link "{prefix+link.name}" '
                    f'when computing swimming height. This can happen '
                    f'when running in headless mode with discardvisual=True '
                    f'and the link only has visual geoms (no collision geoms). '
                    f'Either add a collision geom to this link or set '
                    f'fluid_interaction=false for it.'
                )
            link_heights.append(matching_heights[0])
        self.heights = np.array(link_heights, dtype=float)/self.meters
        self.densities = np.array([link.density for link in links])
        self.buoyancy_centers = np.array([
            np.array(getattr(link, 'buoyancy_center', [0.0, 0.0, 0.0]))
            for link in links
        ], dtype=float)
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
        coefficients_array = np.asarray(self.links_coefficients)
        if np.any(coefficients_array > 0):
            pylog.warning(
                f'Positive drag coefficient(s) for prefix "{prefix}": '
                f'{coefficients_array.tolist()}. FARMS expects '
                'negative coefficients; positive values are anti-damping.'
            )
        if self.sph:
            self.water._surface = 1e8

    cpdef step(self, double time, unsigned int iteration, double dt=0.0):
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
                        dt=dt,
                        buoyancy_center=self.buoyancy_centers[i],
                        inertia=self.inertias[i],
                    )

    cpdef set_frame(self, int frame):
        """Set frame"""
        self.frame = frame

    cpdef void set_water_velocity(self, DTYPEv1 velocity):
        """Set water velocity"""
        self.water.set_velocity(vx=velocity[0], vy=velocity[1], vz=velocity[2])
