"""Control"""

import pybullet
import numpy as np

class AnimatController:
    """AnimatController"""

    def __init__(self, model, network, joints_order):
        super(AnimatController, self).__init__()
        self.model = model
        self.network = network
        self.positions = None
        self.velocities = None
        self.torques = None
        self.joint_list = joints_order

    def update(self):
        """Step"""
        self.network.control_step()
        self.positions = self.network.get_position_output()
        self.velocities = np.zeros_like(self.positions)  # self.network.get_velocity_output()

    def control(self):
        """Control"""
        self.update()
        pybullet.setJointMotorControlArray(
            self.model,
            self.joint_list,
            pybullet.POSITION_CONTROL,
            targetPositions=self.positions,
            targetVelocities=self.velocities,
            # positionGains=[ctrl["pdf"]["p"] for ctrl in controls],
            # velocityGains=[ctrl["pdf"]["d"] for ctrl in controls],
            # forces=[ctrl["pdf"]["f"] for ctrl in controls]
        )
