"""Control"""

from typing import List, Tuple, Dict, Union
from enum import IntEnum
import numpy as np
import numpy.typing as npt


class ControlType(IntEnum):
    """Control type"""
    POSITION = 0
    VELOCITY = 1
    TORQUE = 2

    @staticmethod
    def to_string(control):
        """To string"""
        return {
            ControlType.POSITION: 'position',
            ControlType.VELOCITY: 'velocity',
            ControlType.TORQUE: 'torque',
        }[control]

    @staticmethod
    def from_string(string):
        """From string"""
        return {
            'position': ControlType.POSITION,
            'velocity': ControlType.VELOCITY,
            'torque': ControlType.TORQUE,
        }[string]

    @staticmethod
    def from_string_list(string_list):
        """From string"""
        return [
            ControlType.from_string(control_string)
            for control_string in string_list
        ]


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
