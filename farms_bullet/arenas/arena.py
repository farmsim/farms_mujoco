"""Arena"""

from .create import create_scene
from ..simulations.element import SimulationElement
import numpy as np
import pybullet
import os
from farms_bullet.experiments.salamander.animat import AnimatLink


class Floor(SimulationElement):
    """Floor"""

    def __init__(self, position):
        super(Floor, self).__init__()
        self._position = position

    def spawn(self):
        """Spawn floor"""
        self._identity = self.from_urdf(
            "plane.urdf",
            basePosition=self._position
        )


class Arena:
    """Documentation for Arena"""

    def __init__(self, elements):
        super(Arena, self).__init__()
        self.elements = elements

    def spawn(self):
        """Spawn"""
        for element in self.elements:
            element.spawn()


class FlooredArena(Arena):
    """Arena with floor"""

    def __init__(self, position=None):
        super(FlooredArena, self).__init__(
            [Floor(position if position is not None else [0, 0, -0.1])]
        )

    @property
    def floor(self):
        """Floor"""
        return self.elements[0]


class ArenaScaffold(FlooredArena):
    """Arena for scaffolding"""

    def spawn(self):
        """Spawn"""
        FlooredArena.spawn(self)
        create_scene(self.floor.identity)


class ArenaExperiment1:
    def __init__(self):
        super(ArenaExperiment1, self).__init__()

    def spawn(self):
        """create the arena for experiment 1"""
        arena_dimensions = [1, 3, 0.1]
        arena_color = [1, 0.3, 0, 1]
        arena_color2 = [1, 0.3, 1, 1]
        arena_angle = np.pi / 15
        base_link = AnimatLink(
            geometry=pybullet.GEOM_BOX,
            size=arena_dimensions,
            mass=0,
            joint_axis=[0, 0, 1],
            color=arena_color
        )
        links = [
            AnimatLink(
                geometry=pybullet.GEOM_BOX,
                size=arena_dimensions,
                mass=0,
                parent=0,
                frame_position=[-2 * arena_dimensions[0], 0, 2 * arena_dimensions[2]],
                frame_orientation=[0, arena_angle, 0],
                joint_axis=[0, 0, 1],
                color=arena_color2
            ),
            AnimatLink(
                geometry=pybullet.GEOM_BOX,
                size=arena_dimensions,
                mass=0,
                parent=1,
                frame_position=[-2 * arena_dimensions[0], 0, 2 * arena_dimensions[2]],
                frame_orientation=[0, 0, 0],
                joint_axis=[0, 0, 1],
                color=arena_color
            )
        ]
        pybullet.createMultiBody(
            baseMass=base_link.mass,
            baseCollisionShapeIndex=base_link.collision,
            baseVisualShapeIndex=base_link.visual,
            basePosition=[0, 0, 0],
            baseOrientation=pybullet.getQuaternionFromEuler([0, 0, 0]),
            linkMasses=[link.mass for link in links],
            linkCollisionShapeIndices=[link.collision for link in links],
            linkVisualShapeIndices=[link.visual for link in links],
            linkPositions=[link.position for link in links],
            linkOrientations=[link.orientation for link in links],
            linkInertialFramePositions=[link.f_position for link in links],
            linkInertialFrameOrientations=[link.f_orientation for link in links],
            linkParentIndices=[link.parent for link in links],
            linkJointTypes=[link.joint_type for link in links],
            linkJointAxis=[link.joint_axis for link in links]
        )
