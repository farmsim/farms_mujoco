"""Control"""

from typing import List, Tuple, Dict, Union
from enum import IntEnum
import pybullet
import numpy as np
import numpy.typing as npt

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
        joints_map = model.joints_map
        if controller.joints_names[ControlType.POSITION]:
            kwargs = {}
            if controller.position_args is not None:
                kwargs['positionGains'] = controller.position_args[0]
                kwargs['velocityGains'] = controller.position_args[1]
                kwargs['targetVelocities'] = controller.position_args[2]
            joints_positions = controller.positions(iteration, time, timestep)
            pybullet.setJointMotorControlArray(
                bodyUniqueId=model.identity(),
                jointIndices=[
                    joints_map[joint]
                    for joint in controller.joints_names[ControlType.POSITION]
                ],
                controlMode=pybullet.POSITION_CONTROL,
                targetPositions=[
                    joints_positions[joint]
                    for joint in controller.joints_names[ControlType.POSITION]
                ],
                forces=controller.max_torques[ControlType.POSITION]*torques,
                **kwargs,
            )
        if controller.joints_names[ControlType.VELOCITY]:
            kwargs = {}
            if controller.velocity_args is not None:
                kwargs['positionGains'] = controller.velocity_args[0]
                kwargs['velocityGains'] = controller.velocity_args[1]
            joints_velocities = controller.velocities(iteration, time, timestep)
            pybullet.setJointMotorControlArray(
                bodyUniqueId=model.identity(),
                jointIndices=[
                    joints_map[joint]
                    for joint in controller.joints_names[ControlType.VELOCITY]
                ],
                controlMode=pybullet.VELOCITY_CONTROL,
                targetVelocities=[
                    joints_velocities[joint]*iseconds
                    for joint in controller.joints_names[ControlType.VELOCITY]
                ],
                forces=controller.max_torques[ControlType.VELOCITY]*torques,
                **kwargs,
            )
        if controller.joints_names[ControlType.TORQUE]:
            joints_torques = controller.torques(iteration, time, timestep)
            pybullet.setJointMotorControlArray(
                bodyUniqueId=model.identity(),
                jointIndices=[
                    joints_map[joint]
                    for joint in controller.joints_names[ControlType.TORQUE]
                ],
                controlMode=pybullet.TORQUE_CONTROL,
                forces=[
                    joints_torques[joint]*torques
                    for joint in controller.joints_names[ControlType.TORQUE]
                ],
            )


class ModelController:
    """ModelController"""

    def __init__(
            self,
            joints_names: Tuple[List[str], ...],
            max_torques: Tuple[npt.NDArray[float], ...],
    ):
        super().__init__()
        self.joints_names = joints_names
        self.max_torques = max_torques
        self.indices: Union[None, Tuple[npt.ArrayLike]] = None
        self.position_args: Union[None, Tuple[npt.ArrayLike]] = None
        self.velocity_args: Union[None, Tuple[npt.ArrayLike]] = None
        control_types = list(ControlType)
        assert len(self.joints_names) == len(control_types)
        assert len(self.max_torques) == len(control_types)

    @staticmethod
    def joints_from_control_types(
            joints_names: List[str],
            joints_control_types: Dict[str, List[ControlType]],
    ) -> Tuple[List[str], ...]:
        """From control types"""
        return tuple(
            [
                joint
                for joint in joints_names
                if control_type in joints_control_types[joint]
            ]
            for control_type in list(ControlType)
        )

    @staticmethod
    def max_torques_from_control_types(
            joints_names: List[str],
            max_torques: Dict[str, float],
            joints_control_types: Dict[str, List[ControlType]],
    ) -> Tuple[npt.NDArray, ...]:
        """From control types"""
        return tuple(
            np.array([
                max_torques[joint]
                for joint in joints_names
                if control_type in joints_control_types[joint]
            ])
            for control_type in list(ControlType)
        )

    @classmethod
    def from_control_types(
            cls,
            joints_names: List[str],
            max_torques: Dict[str, float],
            joints_control_types: Dict[str, List[ControlType]],
    ):
        """From control types"""
        return cls(
            joints_names=cls.joints_from_control_types(
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
            for joint in self.joints_names[ControlType.POSITION]
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
            for joint in self.joints_names[ControlType.VELOCITY]
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
            for joint in self.joints_names[ControlType.TORQUE]
        }
