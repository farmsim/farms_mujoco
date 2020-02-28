"""Cython sensors"""

import numpy as np
cimport numpy as np

import pybullet


from ..animats.data.array import NetworkArray3D


class Sensor:
    """Sensor base class for simulation elements"""
    def __init__(self, shape):
        super(Sensor, self).__init__()
        self.array = np.zeros(shape)

    def update(self, iteration):
        """Update"""


cdef class ContactsSensors(NetworkArray3D):
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


cdef class ContactTarget(dict):
    """Documentation for ContactTarget"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, identity, link):
        super(ContactTarget, self).__init__()
        self.identity = identity
        self.link = link


class ContactSensor(Sensor):
    """Model sensors"""

    def __init__(self, n_iterations, animat_id, animat_link, target=None):
        super(ContactSensor, self).__init__([n_iterations, 6])
        self.animat_id = animat_id
        self.animat_link = animat_link
        self.target = target

    def update(self, iteration):
        """Update sensors"""
        self._contacts = pybullet.getContactPoints(
            bodyA=self.animat_id,
            linkIndexA=self.animat_link
        ) if self.target is None else pybullet.getContactPoints(
            bodyA=self.animat_id,
            bodyB=self.target.identity,
            linkIndexA=self.animat_link,
            linkIndexB=self.target.link
        )
        self.array[iteration] = self.get_total_forces()

    def total_force(self, iteration):
        """Toral force"""
        return self.array[iteration, :3] + self.array[iteration, 3:]

    def get_total_forces(self):
        """Get force"""
        return np.sum(
            [
                [
                    # Collision normal reaction
                    contact[9]*contact[7][0],
                    contact[9]*contact[7][1],
                    contact[9]*contact[7][2],
                    # Lateral friction dir 1 + Lateral friction dir 2
                    contact[10]*contact[11][0]+contact[12]*contact[13][0],
                    contact[10]*contact[11][1]+contact[12]*contact[13][1],
                    contact[10]*contact[11][2]+contact[12]*contact[13][2]
                ]
                for contact in self._contacts
            ],
            axis=0
        ) if self._contacts else np.zeros(6)


class JointsStatesSensor(NetworkArray3D):
    """Joint state sensor"""

    def __init__(self, array, model_id, joints, units, enable_ft=False):
        super(JointsStatesSensor, self).__init__(array)
        self._model_id = model_id
        self._joints = joints
        self._enable_ft = enable_ft
        self.units = units
        if self._enable_ft:
            for joint in self._joints:
                pybullet.enableJointForceTorqueSensor(
                    self._model_id,
                    joint
                )

    def update(self, iteration):
        """Update sensor"""
        seconds = self.units.seconds
        inewtons = self.units.newtons
        itorques = self.units.torques
        self.array[iteration] = np.array([
            (
                # Position
                state[0],
                # Velocity
                state[1]*seconds,
                # Forces
                state[2][0]*inewtons,
                state[2][1]*inewtons,
                state[2][2]*inewtons,
                # Torques
                state[2][3]*itorques,
                state[2][4]*itorques,
                state[2][5]*itorques,
                # Motor torque
                state[3]*itorques
            )
            for joint_i, state in enumerate(
                pybullet.getJointStates(self._model_id, self._joints)
            )
        ])


cdef class LinksStatesSensor(NetworkArray3D):
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
        for _, link_i, link_id in links:
            # Collect data
            if link_id == -1:
                # Base link
                base_info = pybullet.getBasePositionAndOrientation(self.animat)
                pos_com = base_info[0]
                ori_com = base_info[1]
                transform_loc = (
                    pybullet.getDynamicsInfo(self.animat, link_id)[3:5]
                )
                transform_inv = pybullet.invertTransform(
                    position=transform_loc[0],
                    orientation=transform_loc[1]
                )
                transform_urdf = pybullet.multiplyTransforms(
                    pos_com,
                    ori_com,
                    transform_inv[0],
                    transform_inv[1]
                )
                pos_urdf = transform_urdf[0]
                ori_urdf = transform_urdf[1]
                base_velocity = pybullet.getBaseVelocity(self.animat)
                lin_velocity = base_velocity[0]
                ang_velocity = base_velocity[1]
            else:
                # Children links
                link_state = pybullet.getLinkState(
                    self.animat,
                    link_id,
                    computeLinkVelocity=1,
                    computeForwardKinematics=1
                )
                pos_com = link_state[0]  # Position of CoM
                ori_com = link_state[1]  # Orientation of CoM
                pos_urdf = link_state[4]  # Position of URDF frame
                ori_urdf = link_state[5]  # Orientation of URDF frame
                lin_velocity = link_state[6]  # Velocity of CoM
                ang_velocity = link_state[7]  # Angular velocity of CoM
            # Position of CoM
            self.array[iteration, link_i, 0] = pos_com[0]*imeters
            self.array[iteration, link_i, 1] = pos_com[1]*imeters
            self.array[iteration, link_i, 2] = pos_com[2]*imeters
            # Orientation of CoM
            self.array[iteration, link_i, 3] = ori_com[0]
            self.array[iteration, link_i, 4] = ori_com[1]
            self.array[iteration, link_i, 5] = ori_com[2]
            self.array[iteration, link_i, 6] = ori_com[3]
            # Position of URDF frame
            self.array[iteration, link_i, 7] = pos_urdf[0]*imeters
            self.array[iteration, link_i, 8] = pos_urdf[1]*imeters
            self.array[iteration, link_i, 9] = pos_urdf[2]*imeters
            # Orientation of URDF frame
            self.array[iteration, link_i, 10] = ori_urdf[0]
            self.array[iteration, link_i, 11] = ori_urdf[1]
            self.array[iteration, link_i, 12] = ori_urdf[2]
            self.array[iteration, link_i, 13] = ori_urdf[3]
            # Velocity of CoM
            self.array[iteration, link_i, 14] = lin_velocity[0]*ivelocity
            self.array[iteration, link_i, 15] = lin_velocity[1]*ivelocity
            self.array[iteration, link_i, 16] = lin_velocity[2]*ivelocity
            # Angular velocity of CoM
            self.array[iteration, link_i, 17] = ang_velocity[0]*seconds
            self.array[iteration, link_i, 18] = ang_velocity[1]*seconds
            self.array[iteration, link_i, 19] = ang_velocity[2]*seconds


class LinkStateSensor(Sensor):
    """Links states sensor"""

    def __init__(self, n_iterations, model_id, link):
        super(LinkStateSensor, self).__init__([n_iterations, 13])
        self._model_id = model_id
        self._link = link

    def update(self, iteration):
        """Update sensor"""
        self.array[iteration] = np.concatenate(
            pybullet.getLinkState(
                bodyUniqueId=self._model_id,
                linkIndex=self._link,
                computeLinkVelocity=1,
                computeForwardKinematics=1
            )[4:]
        )


cdef class Sensors(dict):
    """Sensors"""

    def add(self, new_dict):
        """Add sensors"""
        dict.update(self, new_dict)

    def update(self, iteration):
        """Update all sensors"""
        for sensor in self.values():
            sensor.update(iteration)
