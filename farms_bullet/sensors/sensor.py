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
        self._data[iteration] = np.concatenate(
            [self.get_normal_force(), self.get_lateral_friction()],
            axis=0
        )
        # print("Updating contact {}: {}".format(
        #     self.animat_link,
        #     self._data[iteration]
        # ))

    def total_force(self, iteration):
        """Toral force"""
        return self.data[iteration, :3] + self.data[iteration, 3:]

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
        if self._enable_ft:
            for joint in self._joints:
                pybullet.enableJointForceTorqueSensor(
                    self._model_id,
                    joint
                )

    def update(self, iteration):
        """Update sensor"""
        self._data[iteration] = np.array([
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
        self._data[iteration] = np.concatenate(
            pybullet.getLinkState(
                bodyUniqueId=self._model_id,
                linkIndex=self._link,
                computeLinkVelocity=1,
                computeForwardKinematics=1
            )[4:]
        )


class Sensors(dict):
    """Sensors"""

    def add(self, new_dict):
        """Add sensors"""
        dict.update(self, new_dict)

    def update(self, iteration):
        """Update all sensors"""
        for sensor in self.values():
            sensor.update(iteration)
