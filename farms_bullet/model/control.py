"""Control"""

import pybullet
import numpy as np


def reset_controllers(identity):
    """Reset controllers"""
    n_joints = pybullet.getNumJoints(identity)
    joints = np.arange(n_joints)
    zeros = np.zeros_like(joints)
    pybullet.setJointMotorControlArray(
        identity,
        joints,
        pybullet.POSITION_CONTROL,
        targetPositions=zeros,
        targetVelocities=zeros,
        forces=zeros
    )
    pybullet.setJointMotorControlArray(
        identity,
        joints,
        pybullet.VELOCITY_CONTROL,
        targetVelocities=zeros,
        forces=zeros,
    )
    pybullet.setJointMotorControlArray(
        identity,
        joints,
        pybullet.TORQUE_CONTROL,
        forces=zeros
    )


def control_models(iteration, models, torques, max_torque=100):
    """Control"""
    for model in models:
        if model.controller is None:
            continue
        if iteration == 0:
            reset_controllers(model.identity())
        if model.controller.use_position:
            positions = model.controller.positions(iteration)
            pybullet.setJointMotorControlArray(
                model.identity(),
                model.joints_identities(),
                pybullet.POSITION_CONTROL,
                targetPositions=positions,
                forces=np.repeat(max_torque, len(positions))
            )
        if model.controller.use_torque:
            pybullet.setJointMotorControlArray(
                model.identity(),
                model.joints_identities(),
                pybullet.TORQUE_CONTROL,
                forces=model.controller.torques(iteration)*torques,
            )


class ModelController:
    """ModelController"""

    def __init__(self, joints, use_position, use_torque):
        super(ModelController, self).__init__()
        self.joints = joints  # List of joint names
        self.use_position = use_position
        self.use_torque = use_torque

    def step(self, iteration, time, timestep):
        """Step"""

    def positions(self, iteration):
        """Positions"""
        assert iteration >= 0
        return np.zeros_like(self.joints)

    def velocities(self, iteration):
        """Velocities"""
        assert iteration >= 0
        return np.zeros_like(self.joints)

    def torques(self, iteration):
        """Torques"""
        assert iteration >= 0
        return np.zeros_like(self.joints)
