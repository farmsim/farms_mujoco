"""Sensors"""

import numpy as np

import pybullet


class ModelSensors:
    """Model sensors"""

    def __init__(self, salamander, n_iterations):  # , sensors
        super(ModelSensors, self).__init__()
        # self.sensors = sensors
        # Contact sensors
        self.feet = salamander.feet
        self.contact_forces = np.zeros([n_iterations, 4])
        self.plane = None

        # Force-torque sensors
        self.feet_ft = np.zeros([n_iterations, 4, 6])
        self.joints_sensors = [
            "joint_link_leg_0_L_2",
            "joint_link_leg_0_R_2",
            "joint_link_leg_1_L_2",
            "joint_link_leg_1_R_2"
        ]
        for joint in self.joints_sensors:
            pybullet.enableJointForceTorqueSensor(
                salamander.identity,
                salamander.joints[joint]
            )

    def update(self, iteration, identity, links, joints):
        """Update sensors"""
        self.update_contacts(iteration, identity, links)
        self.update_joints(iteration, identity, joints)

    def update_contacts(self, iteration, identity, links):
        """Update contact sensors"""
        _, self.contact_forces[iteration] = (
            self.get_links_contacts(identity, links, self.plane)
        )

    def update_joints(self, iteration, identity, joints):
        """Update force-torque sensors"""
        self.feet_ft[iteration] = (
            self.get_joints_force_torque(identity, joints)
        )

    @staticmethod
    def get_links_contacts(identity, links, ground):
        """Contacts"""
        contacts = [
            pybullet.getContactPoints(identity, ground, link, -1)
            for link in links
        ]
        forces = [
            np.sum([contact[9] for contact in contacts[link_i]])
            if contacts
            else 0
            for link_i, _ in enumerate(links)
        ]
        return contacts, forces

    @staticmethod
    def get_joints_force_torque(identity, joints):
        """Force-torque on joints"""
        return [
            pybullet.getJointState(identity, joint)[2]
            for joint in joints
        ]
