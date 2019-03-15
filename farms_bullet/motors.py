"""Motors"""

import numpy as np

import pybullet


class ModelMotors:
    """Model motors"""

    def __init__(self):
        super(ModelMotors, self).__init__()
        # Commands
        self.joints_commanded_body = [
            "joint_link_body_{}".format(joint_i+1)
            for joint_i in range(11)
        ]
        self.joints_commanded_legs = [
            "joint_link_leg_{}_{}_{}".format(leg_i, side, joint_i)
            for leg_i in range(2)
            for side in ["L", "R"]
            for joint_i in range(3)
        ]
        self.joints_cmds_body = np.zeros(len(self.joints_commanded_body))
        self.joints_cmds_legs = np.zeros(len(self.joints_commanded_legs))

    def update(self, identity, joints_body, joints_legs):
        """Update"""
        self.update_body(identity, joints_body)
        self.update_legs(identity, joints_legs)

    def update_body(self, identity, joints):
        """Update"""
        self.joints_cmds_body = (
            self.get_joints_commands(identity, joints)
        )

    def update_legs(self, identity, joints):
        """Update"""
        self.joints_cmds_legs = (
            self.get_joints_commands(identity, joints)
        )

    @staticmethod
    def get_joints_commands(identity, joints):
        """Force-torque on joints"""
        return [
            pybullet.getJointState(identity, joint)[3]
            for joint in joints
        ]
