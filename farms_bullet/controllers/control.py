"""Control"""

import pybullet
import numpy as np

class AnimatController:
    """AnimatController"""

    def __init__(self, model, network, joints_order, units):
        super(AnimatController, self).__init__()
        self.model = model
        self.network = network
        self.positions = None
        self.velocities = None
        self.joint_list = joints_order
        self.torques = np.zeros_like(self.joint_list)
        self.units = units
        pybullet.setJointMotorControlArray(
            self.model,
            self.joint_list,
            pybullet.POSITION_CONTROL,
            forces=np.zeros_like(self.joint_list)
        )
        pybullet.setJointMotorControlArray(
            self.model,
            self.joint_list,
            pybullet.VELOCITY_CONTROL,
            forces=np.zeros_like(self.joint_list)
        )
        pybullet.setJointMotorControlArray(
            self.model,
            self.joint_list,
            pybullet.TORQUE_CONTROL,
            forces=np.zeros_like(self.joint_list)
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
        pybullet.setJointMotorControlArray(
            self.model,
            self.joint_list,
            pybullet.POSITION_CONTROL,
            targetPositions=self.positions,
            targetVelocities=self.velocities*self.units.seconds,
            # positionGains=[ctrl["pdf"]["p"] for ctrl in controls],
            # velocityGains=[ctrl["pdf"]["d"] for ctrl in controls],
            # forces=[ctrl["pdf"]["f"] for ctrl in controls]
        )
        # pybullet.setJointMotorControlArray(
        #     self.model,
        #     self.joint_list,
        #     pybullet.TORQUE_CONTROL,
        #     forces=self.torques,
        # )
