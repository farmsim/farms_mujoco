"""Cython sensors"""

import numpy as np
cimport numpy as np

import pybullet


cdef class Sensors(dict):
    """Sensors"""

    def add(self, new_dict):
        """Add sensors"""
        dict.update(self, new_dict)

    def update(self, iteration):
        """Update all sensors"""
        for sensor in self.values():
            sensor.update(iteration)


cdef class ContactsSensors(DoubleArray3D):
    """Model sensors"""

    def __init__(self, array, animat_ids, animat_links, newtons=1):
        super(ContactsSensors, self).__init__(array)
        self.animat_ids = np.array(animat_ids, dtype=np.uintc)
        self.animat_links = np.array(animat_links, dtype=np.intc)
        self.n_sensors = len(animat_links)
        self._contacts = [None for _ in range(self.n_sensors)]
        self.inewtons = 1./newtons

    def update(self, iteration):
        """Update sensors"""
        cdef unsigned int sensor
        for sensor in range(self.n_sensors):
            self._contacts[sensor] = pybullet.getContactPoints(
                bodyA=self.animat_ids[sensor],
                linkIndexA=self.animat_links[sensor]
            )
            if self._contacts[sensor]:
                self._set_contact_forces(
                    iteration,
                    sensor,
                    np.sum(
                        [
                            [
                                # Collision normal reaction
                                contact[9]*contact[7][0]*self.inewtons,
                                contact[9]*contact[7][1]*self.inewtons,
                                contact[9]*contact[7][2]*self.inewtons,
                                # Lateral friction dir 1
                                # + Lateral friction dir 2
                                contact[10]*contact[11][0]*self.inewtons
                                + contact[12]*contact[13][0]*self.inewtons,
                                contact[10]*contact[11][1]*self.inewtons
                                + contact[12]*contact[13][1]*self.inewtons,
                                contact[10]*contact[11][2]*self.inewtons
                                + contact[12]*contact[13][2]*self.inewtons
                            ]
                            for contact in self._contacts[sensor]
                        ],
                        axis=0,
                        dtype=np.float64
                    )
                )

    cdef void _set_contact_forces(
        self,
        unsigned int iteration,
        unsigned int sensor,
        double[:] contact
    ):
        """Set force"""
        cdef unsigned int dim
        for i in range(6):
            self.array[iteration, sensor, i] = contact[i]
        self._set_total_force(iteration, sensor)

    cdef void _set_total_force(
        self,
        unsigned int iteration,
        unsigned int sensor
    ):
        """Set toral force"""
        cdef unsigned int dim
        for dim in range(3):
            self.array[iteration, sensor, 6+dim] = (
                self.array[iteration, sensor, dim]
                + self.array[iteration, sensor, 3+dim]
            )


cdef class JointsStatesSensor(DoubleArray3D):
    """Joint state sensor"""

    def __init__(self, array, model_id, joints, units, enable_ft=True):
        super(JointsStatesSensor, self).__init__(array)
        self.model_id = model_id
        self.joints_map = joints
        self.seconds = units.seconds
        self.inewtons = 1./units.newtons
        self.itorques = 1./units.torques
        if enable_ft:
            for joint in self.joints_map:
                pybullet.enableJointForceTorqueSensor(
                    self.model_id,
                    joint
                )

    cpdef tuple get_joints_states(self):
        """Get joints states"""
        return pybullet.getJointStates(self.model_id, self.joints_map)

    cpdef void update(self, unsigned int iteration):
        """Update sensor"""
        cdef unsigned int joint_i
        cdef double position, velocity, torque, fx, fy, fz, tx, ty, tz
        for (
                joint_i,
                (position, velocity, (fx, fy, fz, tx, ty, tz), torque),
        ) in enumerate(self.get_joints_states()):
            # Position
            self.array[iteration, joint_i, JOINT_POSITION] = position
            # Velocity
            self.array[iteration, joint_i, JOINT_VELOCITY] = velocity*self.seconds
            # Forces
            self.array[iteration, joint_i, JOINT_FORCE_X] = fx*self.inewtons
            self.array[iteration, joint_i, JOINT_FORCE_Y] = fy*self.inewtons
            self.array[iteration, joint_i, JOINT_FORCE_Z] = fz*self.inewtons
            # Torques
            self.array[iteration, joint_i, JOINT_TORQUE_X] = tx*self.itorques
            self.array[iteration, joint_i, JOINT_TORQUE_X] = ty*self.itorques
            self.array[iteration, joint_i, JOINT_TORQUE_Z] = tz*self.itorques
            # Motor torque
            self.array[iteration, joint_i, JOINT_TORQUE] = torque*self.itorques


cdef class LinksStatesSensor(DoubleArray3D):
    """Links states sensor

    links is an array of size (N, 3) where the 3 values are:
    [
        link_name,  # Name of the link
        link_i,  # Index in table
        link_id  # Index in animat
    ]
    """

    def __init__(self, array, animat_id, links, units):
        super(LinksStatesSensor, self).__init__(array)
        self.animat = animat_id
        self.links = links
        self.units = units

    def update(self, iteration):
        """Update sensor"""
        self.collect(iteration, self.links)

    cpdef object get_base_link_state(self):
        """Get link states"""
        base_info = pybullet.getBasePositionAndOrientation(self.animat)
        pos_com = base_info[0]
        ori_com = base_info[1]
        transform_loc = (
            pybullet.getDynamicsInfo(self.animat, -1)[3:5]
        )
        transform_inv = pybullet.invertTransform(
            position=transform_loc[0],
            orientation=transform_loc[1],
        )
        transform_urdf = pybullet.multiplyTransforms(
            pos_com,
            ori_com,
            transform_inv[0],
            transform_inv[1],
        )
        base_velocity = pybullet.getBaseVelocity(self.animat)
        return (
            pos_com,
            ori_com,
            transform_urdf[0],
            transform_urdf[1],
            base_velocity[0],
            base_velocity[1],
        )

    cpdef object get_children_links_states(self):
        """Get link states"""
        return pybullet.getLinkStates(
            self.animat,
            self.links,
            computeLinkVelocity=1,
            computeForwardKinematics=1
        )

    cpdef void collect(self, unsigned int iteration, object links):
        """Collect gps data"""
        cdef int link_id
        cdef unsigned int link_i
        cdef double pos_com[3]
        cdef double ori_com[4]
        cdef double pos_urdf[3]
        cdef double ori_urdf[4]
        cdef double lin_velocity[3]
        cdef double ang_velocity[3]
        cdef double imeters = 1./self.units.meters
        cdef double ivelocity = 1./self.units.velocity
        cdef double seconds = self.units.seconds
        states = self.get_children_links_states()
        for link_i, link_id in enumerate(links):
            # Collect data
            if link_id == -1:
                # Base link
                link_state = self.get_base_link_state()
                pos_com = link_state[0]
                ori_com = link_state[1]
                pos_urdf = link_state[2]
                ori_urdf = link_state[3]
                lin_velocity = link_state[4]
                ang_velocity = link_state[5]
            else:
                # Children links
                link_state = states[link_i]
                pos_com = link_state[0]  # Position of CoM
                ori_com = link_state[1]  # Orientation of CoM
                pos_urdf = link_state[4]  # Position of URDF frame
                ori_urdf = link_state[5]  # Orientation of URDF frame
                lin_velocity = link_state[6]  # Velocity of CoM
                ang_velocity = link_state[7]  # Angular velocity of CoM
            # Position of CoM
            self.array[iteration, link_i, LINK_COM_POSITION_X] = pos_com[0]*imeters
            self.array[iteration, link_i, LINK_COM_POSITION_Y] = pos_com[1]*imeters
            self.array[iteration, link_i, LINK_COM_POSITION_Z] = pos_com[2]*imeters
            # Orientation of CoM
            self.array[iteration, link_i, LINK_COM_ORIENTATION_X] = ori_com[0]
            self.array[iteration, link_i, LINK_COM_ORIENTATION_Y] = ori_com[1]
            self.array[iteration, link_i, LINK_COM_ORIENTATION_Z] = ori_com[2]
            self.array[iteration, link_i, LINK_COM_ORIENTATION_W] = ori_com[3]
            # Position of URDF frame
            self.array[iteration, link_i, LINK_URDF_POSITION_X] = pos_urdf[0]*imeters
            self.array[iteration, link_i, LINK_URDF_POSITION_Y] = pos_urdf[1]*imeters
            self.array[iteration, link_i, LINK_URDF_POSITION_Z] = pos_urdf[2]*imeters
            # Orientation of URDF frame
            self.array[iteration, link_i, LINK_URDF_ORIENTATION_X] = ori_urdf[0]
            self.array[iteration, link_i, LINK_URDF_ORIENTATION_Y] = ori_urdf[1]
            self.array[iteration, link_i, LINK_URDF_ORIENTATION_Z] = ori_urdf[2]
            self.array[iteration, link_i, LINK_URDF_ORIENTATION_W] = ori_urdf[3]
            # Velocity of CoM
            self.array[iteration, link_i, LINK_COM_VELOCITY_LIN_X] = lin_velocity[0]*ivelocity
            self.array[iteration, link_i, LINK_COM_VELOCITY_LIN_Y] = lin_velocity[1]*ivelocity
            self.array[iteration, link_i, LINK_COM_VELOCITY_LIN_Z] = lin_velocity[2]*ivelocity
            # Angular velocity of CoM
            self.array[iteration, link_i, LINK_COM_VELOCITY_ANG_X] = ang_velocity[0]*seconds
            self.array[iteration, link_i, LINK_COM_VELOCITY_ANG_Y] = ang_velocity[1]*seconds
            self.array[iteration, link_i, LINK_COM_VELOCITY_ANG_Z] = ang_velocity[2]*seconds
