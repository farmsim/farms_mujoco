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


def control_models(iteration, models, seconds, torques, max_torque=100):
    """Control"""
    # if not all(np.abs(velocities) < 2*np.pi*3):
    #     print("Velocities too fast:\n{}".format(velocities/(2*np.pi)))
    # isec = 1.0/seconds
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
                # positionGains=1e0*np.ones_like(positions),
                # velocityGains=1e0*np.ones_like(positions),
                # forces=1e6*np.ones_like(positions)
                # targetVelocities=model.controller.velocities(iteration)*isec,
                # forces=positions*1e1
                # targetVelocities=velocities/units.seconds,
                # targetVelocities=np.zeros_like(positions),
                # positionGains=[ctrl["pdf"]["p"] for ctrl in controls],
                # velocityGains=[ctrl["pdf"]["d"] for ctrl in controls],
                # positionGains=gain_position,
                # velocityGains=gain_velocity,
                # forces=[ctrl["pdf"]["f"] for ctrl in controls],
                # maxVelocities=2*np.pi*0.1*np.ones_like(positions)
            )
        if model.controller.use_torque:
            pybullet.setJointMotorControlArray(
                model.identity(),
                model.joints_identities(),
                pybullet.TORQUE_CONTROL,
                forces=model.controller.torques(iteration)*torques,
                # forces=positions*1e1
                # targetVelocities=velocities/units.seconds,
                # targetVelocities=np.zeros_like(positions),
                # positionGains=[ctrl["pdf"]["p"] for ctrl in controls],
                # velocityGains=[ctrl["pdf"]["d"] for ctrl in controls],
                # positionGains=gain_position,
                # velocityGains=gain_velocity,
                # forces=[ctrl["pdf"]["f"] for ctrl in controls],
                # maxVelocities=2*np.pi*0.1*np.ones_like(positions)
            )
        # for joint_i, joint in enumerate(joint_list):
        #     pybullet.setJointMotorControl2(
        #         model.identity(),
        #         joint,
        #         pybullet.POSITION_CONTROL,
        #         targetPosition=positions[joint_i],
        #         targetVelocity=velocities[joint_i]*units.seconds,
        #         # positionGains=[ctrl["pdf"]["p"] for ctrl in controls],
        #         # velocityGains=[ctrl["pdf"]["d"] for ctrl in controls],
        #         # forces=[ctrl["pdf"]["f"] for ctrl in controls],
        #         maxVelocity=2*np.pi*3
        #     )
        # pybullet.setJointMotorControlArray(
        #     model.identity(),
        #     joint_list,
        #     pybullet.TORQUE_CONTROL,
        #     forces=torques*units.torques,
        # )


class ModelController:
    """ModelController"""

    def __init__(self, joints, use_position, use_torque):
        super(ModelController, self).__init__()
        self.joints = joints  # List of joint names
        self.use_position = use_position
        self.use_torque = use_torque

    def step(self):
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
