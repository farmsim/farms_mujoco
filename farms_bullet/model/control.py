"""Control"""

from typing import List, Dict
from enum import IntEnum
import pybullet
import numpy as np

from farms_data.units import SimulationUnitScaling
from .model import SimulationModels


class ControlType(IntEnum):
    """Control type"""
    POSITION = 0
    VELOCITY = 1
    TORQUE = 2


def reset_controllers(identity: int):
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


def control_models(
        iteration: int,
        time: float,
        timestep: float,
        models: SimulationModels,
        units: SimulationUnitScaling,
):
    """Control"""
    torques = units.torques
    iseconds = 1/units.seconds
    for model in models:
        if model.controller is None:
            continue
        controller = model.controller
        if controller.joints[ControlType.POSITION]:
            joints_positions = controller.positions(iteration, time, timestep)
            kwargs = {}
            if isinstance(joints_positions, tuple):
                # Gains are provided
                (
                    joints_positions,
                    kwargs['positionGains'],
                    kwargs['velocityGains'],
                ) = joints_positions
                kwargs['targetVelocities'] = (
                    np.zeros_like(kwargs['velocityGains'])
                )
            pybullet.setJointMotorControlArray(
                bodyUniqueId=model.identity(),
                jointIndices=[
                    model.joints_map[joint]
                    for joint in controller.joints[ControlType.POSITION]
                ],
                controlMode=pybullet.POSITION_CONTROL,
                targetPositions=[
                    joints_positions[joint]
                    for joint in controller.joints[ControlType.POSITION]
                ],
                forces=controller.max_torques[ControlType.POSITION]*torques,
                **kwargs,
            )
        if controller.joints[ControlType.VELOCITY]:
            joints_velocities = controller.velocities(iteration, time, timestep)
            kwargs = {}
            if isinstance(joints_velocities, tuple):
                # Gains are provided
                (
                    joints_velocities,
                    kwargs['positionGains'],
                    kwargs['velocityGains'],
                ) = joints_velocities
            pybullet.setJointMotorControlArray(
                bodyUniqueId=model.identity(),
                jointIndices=[
                    model.joints_map[joint]
                    for joint in controller.joints[ControlType.VELOCITY]
                ],
                controlMode=pybullet.VELOCITY_CONTROL,
                targetVelocities=[
                    joints_velocities[joint]*iseconds
                    for joint in controller.joints[ControlType.VELOCITY]
                ],
                forces=controller.max_torques[ControlType.VELOCITY]*torques,
                **kwargs,
            )
        if controller.joints[ControlType.TORQUE]:
            joints_torques = controller.torques(iteration, time, timestep)
            pybullet.setJointMotorControlArray(
                bodyUniqueId=model.identity(),
                jointIndices=[
                    model.joints_map[joint]
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

    def __init__(self, joints: List[List[str]], max_torques: List[List[float]]):
        super().__init__()
        self.joints = joints
        self.max_torques = max_torques
        control_types = list(ControlType)
        assert len(self.joints) == len(control_types)
        assert len(self.max_torques) == len(control_types)

    @staticmethod
    def joints_from_control_types(
            joints_names: List[str],
            joints_control_types: Dict[str, ControlType],
    ):
        """From control types"""
        return [
            [
                joint
                for joint in joints_names
                if joints_control_types[joint] == control_type
            ]
            for control_type in list(ControlType)
        ]

    @staticmethod
    def max_torques_from_control_types(
            joints_names: List[str],
            max_torques: Dict[str, float],
            joints_control_types: Dict[str, ControlType],
    ):
        """From control types"""
        return [
            np.array([
                max_torques[joint]
                for joint in joints_names
                if joints_control_types[joint] == control_type
            ])
            for control_type in list(ControlType)
        ]

    @classmethod
    def from_control_types(
            cls,
            joints_names: List[str],
            max_torques: Dict[str, float],
            joints_control_types: Dict[str, ControlType],
    ):
        """From control types"""
        return cls(
            joints=cls.joints_from_control_types(
                joints_names=joints_names,
                joints_control_types=joints_control_types,
            ),
            max_torques=cls.max_torques_from_control_types(
                joints_names=joints_names,
                max_torques=max_torques,
                joints_control_types=joints_control_types,
            ),
        )

    def step(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ):
        """Step"""

    def positions(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> Dict[str, float]:
        """Positions"""
        assert iteration >= 0
        assert time >= 0
        assert timestep > 0
        return {
            joint: 0
            for joint in self.joints[ControlType.POSITION]
        }

    def velocities(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> Dict[str, float]:
        """Velocities"""
        assert iteration >= 0
        assert time >= 0
        assert timestep > 0
        return {
            joint: 0
            for joint in self.joints[ControlType.VELOCITY]
        }

    def torques(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> Dict[str, float]:
        """Torques"""
        assert iteration >= 0
        assert time >= 0
        assert timestep > 0
        return {
            joint: 0
            for joint in self.joints[ControlType.TORQUE]
        }
