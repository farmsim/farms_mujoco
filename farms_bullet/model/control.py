"""Control"""

from enum import IntEnum
import pybullet
import numpy as np


class ControlType(IntEnum):
    """Control type"""
    POSITION = 0
    VELOCITY = 1
    TORQUE = 2


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


def control_models(iteration, models, torques):
    """Control"""
    for model in models:
        if model.controller is None:
            continue
        if iteration == 0:
            reset_controllers(model.identity())
        controller = model.controller
        if controller.joints[ControlType.POSITION]:
            joints_positions = controller.positions(iteration)
            pybullet.setJointMotorControlArray(
                bodyUniqueId=model.identity(),
                jointIndices=[
                    model._joints[joint]
                    for joint in controller.joints[ControlType.POSITION]
                ],
                controlMode=pybullet.POSITION_CONTROL,
                targetPositions=[
                    joints_positions[joint]
                    for joint in controller.joints[ControlType.POSITION]
                ],
                forces=controller.max_torques[ControlType.POSITION]*torques,
            )
        if controller.joints[ControlType.TORQUE]:
            joints_torques = controller.torques(iteration)
            pybullet.setJointMotorControlArray(
                bodyUniqueId=model.identity(),
                jointIndices=[
                    model._joints[joint]
                    for joint in controller.joints[ControlType.TORQUE]
                ],
                controlMode=pybullet.TORQUE_CONTROL,
                forces=np.clip(
                    [
                        joints_torques[joint]
                        for joint in controller.joints[ControlType.TORQUE]
                    ],
                    -controller.max_torques[ControlType.TORQUE],
                    controller.max_torques[ControlType.TORQUE],
                )*torques,
            )


class ModelController:
    """ModelController"""

    def __init__(self, joints, control_types, max_torques):
        super(ModelController, self).__init__()
        self.joints = [
            [
                joint
                for joint in joints
                if control_types[joint] == control_type
            ]
            for control_type in [
                ControlType.POSITION,
                ControlType.VELOCITY,
                ControlType.TORQUE
            ]
        ]
        self.max_torques = [
            np.array([
                max_torques[joint]
                for joint in joints
                if control_types[joint] == control_type
            ])
            for control_type in [
                ControlType.POSITION,
                ControlType.VELOCITY,
                ControlType.TORQUE
            ]
        ]

    def step(self, iteration, time, timestep):
        """Step"""

    def positions(self, iteration):
        """Positions"""
        assert iteration >= 0
        return {
            'joint_{}'.format(joint_i): 0
            for joints in self.joints
        }

    def velocities(self, iteration):
        """Velocities"""
        assert iteration >= 0
        return {
            'joint_{}'.format(joint_i): 0
            for joints in self.joints
        }

    def torques(self, iteration):
        """Torques"""
        assert iteration >= 0
        return {
            'joint_{}'.format(joint_i): 0
            for joints in self.joints
        }
