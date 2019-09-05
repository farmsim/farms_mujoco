"""Arena"""

import os

import numpy as np
import pybullet

from .create import create_scene
from ..simulations.element import SimulationElement
from ..animats.link import AnimatLink


class Floor(SimulationElement):
    """Floor"""

    def __init__(self, position):
        super(Floor, self).__init__()
        self._position = np.array(position)

    def spawn(self):
        """Spawn floor"""
        size = 0.5*np.array([10, 10, 10])
        base_link = AnimatLink(
            geometry=pybullet.GEOM_BOX,
            size=size,
            inertial_position=[0, 0, 0],
            position=[0, 0, 0],
            joint_axis=[0, 0, 1],
            mass=0,
            color=[1, 0, 0, 1],
            # collision_options=collision_options,
            # visual_options=visual_options
        )
        self._identity = pybullet.createMultiBody(
            baseMass=base_link.mass,
            baseCollisionShapeIndex=base_link.collision,
            baseVisualShapeIndex=base_link.visual,
            basePosition=self._position-np.array([0, 0, size[2]]),
            baseOrientation=pybullet.getQuaternionFromEuler([0, 0, 0]),
            baseInertialFramePosition=base_link.inertial_position,
            baseInertialFrameOrientation=base_link.inertial_orientation
        )
        dir_path = os.path.dirname(os.path.realpath(__file__))
        texUid = pybullet.loadTexture(dir_path+"/BIOROB2_blue.png")
        pybullet.changeVisualShape(
            self._identity, -1,
            textureUniqueId=texUid,
            # rgbaColor=[1, 1, 1, 1],
            # specularColor=[1, 1, 1]
        )


class FloorURDF(SimulationElement):
    """Floor"""

    def __init__(self, position):
        super(FloorURDF, self).__init__()
        self._position = position

    def spawn(self):
        """Spawn floor"""
        self._identity = self.from_urdf(
            "plane.urdf",
            basePosition=self._position
        )
        dir_path = os.path.dirname(os.path.realpath(__file__))
        texUid = pybullet.loadTexture(dir_path+"/BIOROB2_blue.png")
        pybullet.changeVisualShape(
            self._identity, -1,
            textureUniqueId=texUid,
            rgbaColor=[1, 1, 1, 1],
            specularColor=[1, 1, 1]
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
            [FloorURDF(position if position is not None else [0, 0, 0])]
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


class Ramp(SimulationElement):
    """Floor"""

    def __init__(self, angle, units):
        super(Ramp, self).__init__()
        self.angle = angle
        self.units = units

    def spawn(self):
        """Spawn floor"""
        ground_dim = [1, 20, 0.1]
        ramp_dim = [3, 20, 0.1]
        upper_lower_dim = [1, 20, 0.1]
        arena_color = [1, 0.8, 0.5, 1]

        # Arena definition
        base_link = AnimatLink(
            geometry=pybullet.GEOM_BOX,
            size=ground_dim,
            mass=0,
            joint_axis=[0, 0, 1],
            color=arena_color,
            units=self.units
        )
        links = [
            AnimatLink(
                geometry=pybullet.GEOM_BOX,
                size=ramp_dim,
                mass=0,
                parent=0,
                frame_position=[
                    (
                        - ground_dim[0]
                        - np.cos(self.angle) * ramp_dim[0]
                    ),
                    0,
                    np.sin(self.angle) * ramp_dim[0]
                ],
                frame_orientation=[0, self.angle, 0],
                joint_axis=[0, 0, 1],
                color=arena_color,
                units=self.units
            ),
            AnimatLink(
                geometry=pybullet.GEOM_BOX,
                size=ground_dim,
                mass=0,
                parent=1,
                frame_position=[
                    (
                        - ground_dim[0]
                        - 2 * np.cos(self.angle) * ramp_dim[0]
                        - upper_lower_dim[0]
                    ),
                    0,
                    2 * np.sin(self.angle) * ramp_dim[0]
                ],
                frame_orientation=[0, 0, 0],
                joint_axis=[0, 0, 1],
                color=arena_color,
                units=self.units
            )
        ]

        # Spawn
        self._identity = pybullet.createMultiBody(
            baseMass=base_link.mass,
            baseCollisionShapeIndex=base_link.collision,
            baseVisualShapeIndex=base_link.visual,
            basePosition=[0, 0, -ground_dim[2]],
            baseOrientation=pybullet.getQuaternionFromEuler([0, 0, 0]),
            linkMasses=[link.mass for link in links],
            linkCollisionShapeIndices=[link.collision for link in links],
            linkVisualShapeIndices=[link.visual for link in links],
            linkPositions=[link.position for link in links],
            linkOrientations=[link.orientation for link in links],
            linkInertialFramePositions=[
                link.inertial_position
                for link in links
            ],
            linkInertialFrameOrientations=[
                link.inertial_orientation
                for link in links
            ],
            linkParentIndices=[link.parent for link in links],
            linkJointTypes=[link.joint_type for link in links],
            linkJointAxis=[link.joint_axis for link in links]
        )

        # Dynamics properties
        pybullet.changeDynamics(
            bodyUniqueId=self.identity,
            linkIndex=0,
            lateralFriction=2,
            spinningFriction=0,
            rollingFriction=0,
            contactDamping=1e2,
            contactStiffness=1e4
        )

        # # Texture
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        # texUid = pybullet.loadTexture(dir_path+"/BIOROB2_blue.png")
        # pybullet.changeVisualShape(
        #     self._identity, -1,
        #     textureUniqueId=texUid,
        #     rgbaColor=[1, 1, 1, 1],
        #     specularColor=[1, 1, 1]
        # )


class ArenaRamp(Arena):
    """ArenaRamp"""

    def __init__(self, units, ramp_angle=None, elements=None):
        angle = (
            np.deg2rad(ramp_angle)
            if ramp_angle is not None
            else np.deg2rad(30)
        )
        if elements is None:
            elements = []
        super(ArenaRamp, self).__init__(
            [Ramp(angle=angle, units=units)]
            + elements
        )

    @property
    def floor(self):
        """Floor"""
        return self.elements[0]


class Water(SimulationElement):
    """Floor"""

    def __init__(self, units):
        super(Water, self).__init__()
        self.units = units

    def spawn(self):
        """Spawn floor"""
        water_size = [50, 50, 50]
        water_color = [0.5, 0.5, 0.9, 0.7]

        # Water definition
        base_link = AnimatLink(
            geometry=pybullet.GEOM_BOX,
            size=water_size,
            mass=0,
            joint_axis=[0, 0, 1],
            color=water_color,
            units=self.units
        )

        # Spawn
        self._identity = pybullet.createMultiBody(
            baseMass=base_link.mass,
            baseCollisionShapeIndex=base_link.collision,
            baseVisualShapeIndex=base_link.visual,
            basePosition=[0, 0, -0.1-water_size[2]],
            baseOrientation=pybullet.getQuaternionFromEuler([0, 0, 0]),
        )

        # Dynamics properties
        group = 0  #other objects don't collide with me
        mask = 0  # don't collide with any other object
        pybullet. setCollisionFilterGroupMask(
            self.identity,
            -1,
            group,
            mask
        )


class ArenaWater(ArenaRamp):
    """ArenaRamp"""

    def __init__(self, units, ramp_angle=None, elements=None):
        super(ArenaWater, self).__init__(
            units=units,
            ramp_angle=ramp_angle,
            elements=[Water(units)]
        )

    @property
    def water(self):
        """Floor"""
        return self.elements[1]
