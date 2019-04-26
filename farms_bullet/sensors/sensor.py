"""Sensors"""

import numpy as np
import pybullet


class Sensor:
    """Sensor base class for simulation elements"""
    def __init__(self, shape):
        super(Sensor, self).__init__()
        self._data = np.zeros(shape)

    def update(self, iteration):
        """Update"""

    @property
    def data(self):
        """Sensor data"""
        return self._data


class ContactSensor(Sensor):
    """Model sensors"""

    def __init__(self, n_iterations, animat_id, animat_link, target_id, target_link):
        super(ContactSensor, self).__init__([n_iterations, 6])
        self.animat_id = animat_id
        self.animat_link = animat_link
        self.target_id = target_id
        self.target_link = target_link

    def update(self, iteration):
        """Update sensors"""
        self._contacts = pybullet.getContactPoints(
            self.animat_id,
            self.target_id,
            self.animat_link,
            self.target_link
        )
        self._data[iteration] = np.concatenate(
            [self.get_normal_force(), self.get_lateral_friction()],
            axis=0
        )

    def get_normal_force(self):
        """Get force"""
        return np.sum([
            contact[9]*np.array(contact[7])
            for contact in self._contacts
        ], axis=0) if self._contacts else np.zeros(3)

    def get_lateral_friction(self):
        """Get force"""
        return np.sum([
            contact[10]*np.array(contact[11])  # Lateral friction dir 1
            + contact[12]*np.array(contact[13])  # Lateral friction dir 2
            for contact in self._contacts
        ], axis=0) if self._contacts else np.zeros(3)


class JointsStatesSensor(Sensor):
    """Joint state sensor"""

    def __init__(self, n_iterations, model_id, joints, enable_ft=False):
        super(JointsStatesSensor, self).__init__([n_iterations, len(joints), 9])
        self._model_id = model_id
        self._joints = joints
        self._enable_ft = enable_ft
        self._state = None
        if self._enable_ft:
            for joint in self._joints:
                pybullet.enableJointForceTorqueSensor(
                    self._model_id,
                    joint
                )

    def update(self, iteration):
        """Update sensor"""
        self._state = pybullet.getJointStates(self._model_id, self._joints)
        self._data[iteration] = np.array([
            [
                self._state[joint_i][0],
                self._state[joint_i][1]
            ] + list(self._state[joint_i][2]) + [
                self._state[joint_i][3]
            ]
            for joint_i, state in enumerate(self._state)
        ])


class LinkStateSensor(Sensor):
    """Links states sensor"""

    def __init__(self, n_iterations, model_id, link):
        super(LinkStateSensor, self).__init__([n_iterations, 13])
        self._model_id = model_id
        self._link = link
        self._state = None

    def update(self, iteration):
        """Update sensor"""
        self._state = pybullet.getLinkState(
            bodyUniqueId=self._model_id,
            linkIndex=self._link,
            computeLinkVelocity=1,
            computeForwardKinematics=1
        )
        self._data[iteration] = np.concatenate(self._state[4:])


class Sensors(dict):
    """Sensors"""

    def add(self, new_dict):
        """Add sensors"""
        dict.update(self, new_dict)

    def update(self, iteration):
        """Update all sensors"""
        for sensor in self.values():
            sensor.update(iteration)
