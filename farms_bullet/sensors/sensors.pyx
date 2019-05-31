"""Cython sensors"""

import numpy as np
cimport numpy as np

import pybullet


from ..animats.array import NetworkArray3D


class Sensor:
    """Sensor base class for simulation elements"""
    def __init__(self, shape):
        super(Sensor, self).__init__()
        self.array = np.zeros(shape)

    def update(self, iteration):
        """Update"""


cdef class ContactsSensors(NetworkArray3D):
    """Model sensors"""

    def __init__(self, array, animat_ids, animat_links):
        super(ContactsSensors, self).__init__(array)
        self.animat_ids = np.array(animat_ids, dtype=np.uintc)
        self.animat_links = np.array(animat_links, dtype=np.intc)
        self.n_sensors = len(animat_links)
        self._contacts = [None for _ in range(self.n_sensors)]

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
                                contact[9]*contact[7][0],
                                contact[9]*contact[7][1],
                                contact[9]*contact[7][2],
                                # Lateral friction dir 1
                                # + Lateral friction dir 2
                                contact[10]*contact[11][0]
                                + contact[12]*contact[13][0],
                                contact[10]*contact[11][1]
                                + contact[12]*contact[13][1],
                                contact[10]*contact[11][2]
                                + contact[12]*contact[13][2]
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


class ContactTarget(dict):
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


class JointsStatesSensor(Sensor):
    """Joint state sensor"""

    def __init__(self, n_iterations, model_id, joints, enable_ft=False):
        super(JointsStatesSensor, self).__init__([n_iterations, len(joints), 9])
        self._model_id = model_id
        self._joints = joints
        self._enable_ft = enable_ft
        if self._enable_ft:
            for joint in self._joints:
                pybullet.enableJointForceTorqueSensor(
                    self._model_id,
                    joint
                )

    def update(self, iteration):
        """Update sensor"""
        self.array[iteration] = np.array([
            (state[0], state[1]) + state[2] + (state[3],)
            for joint_i, state in enumerate(
                pybullet.getJointStates(self._model_id, self._joints)
            )
        ])


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
