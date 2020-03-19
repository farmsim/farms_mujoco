"""Control"""

import pybullet
import numpy as np


class ModelController:
    """ModelController"""

    def __init__(self, identity, network, joints_order, units):
        super(ModelController, self).__init__()
        self.identity = identity
        self.network = network
        self.joint_list = joints_order
        n_joints = len(self.joint_list)
        # Commands
        self.positions = None
        self.velocities = None
        self.torques = np.zeros(n_joints)
        # Units
        self.units = units
        self.unit_iseconds = 1./units.seconds
        # Gains
        self.gain_position = 1e-1*np.ones(n_joints)*(
            self.units.torques
        )
        self.gain_velocity = 1e0*np.ones(n_joints)*(
            self.units.torques*self.units.seconds
        )
        self.reset()

    def reset(self):
        """Reset controllers"""
        # Reset controllers
        zeros = np.zeros_like(self.joint_list)
        pybullet.setJointMotorControlArray(
            self.identity,
            self.joint_list,
            pybullet.POSITION_CONTROL,
            forces=zeros
        )
        pybullet.setJointMotorControlArray(
            self.identity,
            self.joint_list,
            pybullet.VELOCITY_CONTROL,
            forces=zeros
        )
        pybullet.setJointMotorControlArray(
            self.identity,
            self.joint_list,
            pybullet.TORQUE_CONTROL,
            forces=zeros
        )

    def update(self):
        """Step"""
        self.network.control_step()
        self.positions = self.network.get_position_output()
        self.velocities = self.network.get_velocity_output()
        # a_filter = 0
        # self.torques = (
        #     a_filter*self.torques
        #     + (1-a_filter)*self.network.get_torque_output()
        # )

    def control(self):
        """Control"""
        self.update()
        # if not all(np.abs(self.velocities) < 2*np.pi*3):
        #     print("Velocities too fast:\n{}".format(self.velocities/(2*np.pi)))
        pybullet.setJointMotorControlArray(
            self.identity,
            self.joint_list,
            pybullet.POSITION_CONTROL,
            targetPositions=self.positions,
            targetVelocities=self.velocities*self.unit_iseconds,
            # forces=self.positions*1e1
            # targetVelocities=self.velocities/self.units.seconds,
            # targetVelocities=np.zeros_like(self.positions),
            # positionGains=[ctrl["pdf"]["p"] for ctrl in controls],
            # velocityGains=[ctrl["pdf"]["d"] for ctrl in controls],
            # positionGains=self.gain_position,
            # velocityGains=self.gain_velocity,
            # forces=[ctrl["pdf"]["f"] for ctrl in controls],
            # maxVelocities=2*np.pi*0.1*np.ones_like(self.positions)
        )
        # for joint_i, joint in enumerate(self.joint_list):
        #     pybullet.setJointMotorControl2(
        #         self.identity,
        #         joint,
        #         pybullet.POSITION_CONTROL,
        #         targetPosition=self.positions[joint_i],
        #         targetVelocity=self.velocities[joint_i]*self.units.seconds,
        #         # positionGains=[ctrl["pdf"]["p"] for ctrl in controls],
        #         # velocityGains=[ctrl["pdf"]["d"] for ctrl in controls],
        #         # forces=[ctrl["pdf"]["f"] for ctrl in controls],
        #         maxVelocity=2*np.pi*3
        #     )
        # pybullet.setJointMotorControlArray(
        #     self.identity,
        #     self.joint_list,
        #     pybullet.TORQUE_CONTROL,
        #     forces=self.torques*self.units.torques,
        # )
