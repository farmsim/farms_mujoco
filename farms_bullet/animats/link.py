"""Animat link"""

import numpy as np
import trimesh as tri
import pybullet

from ..simulations.simulation_options import SimulationUnitScaling

class AnimatLink(dict):
    """Animat link"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, units, **kwargs):
        super(AnimatLink, self).__init__()
        additional_kwargs = {}
        self.units = units
        self.size = kwargs.pop("size", None)
        if self.size is not None:
            self.size = np.array(self.size)*self.units.meters
        self.radius = kwargs.pop("radius", None)
        if self.radius is not None:
            self.radius = np.array(self.radius)*self.units.meters
        self.height = kwargs.pop("height", None)
        if self.height is not None:
            self.height = np.array(self.height)*self.units.meters
        self.filename = kwargs.pop("filename", None)
        self.mass = kwargs.pop("mass", None)
        if self.mass is not None:
            self.mass = np.array(self.mass)*self.units.kilograms
        self.volume = kwargs.pop("volume", None)
        if self.volume is not None:
            self.volume = np.array(self.volume)*self.units.volume
        self.density = kwargs.pop("density", 1000*self.units.density)
        self.scale = np.array(kwargs.pop("scale", [1.0, 1.0, 1.0]))
        if self.size is not None:
            additional_kwargs["halfExtents"] = self.size
        if self.radius is not None:
            additional_kwargs["radius"] = self.radius
        if self.height is not None:
            additional_kwargs["height"] = self.height
        if self.filename is not None:
            additional_kwargs["fileName"] = self.filename
            if self.mass is None:
                additional_kwargs["meshScale"] = self.scale*self.units.meters
                self.volume = (
                    self.scale[0]*self.scale[1]*self.scale[2]*self.units.volume
                    *tri.load_mesh(self.filename).volume
                )
                self.mass = self.density*self.volume
        self.geometry = kwargs.pop("geometry", pybullet.GEOM_BOX)
        if self.mass is None:
            if self.geometry == pybullet.GEOM_BOX:
                self.volume = self.size[0]*self.size[1]*self.size[2]
            elif self.geometry == pybullet.GEOM_SPHERE:
                self.volume = 4/3*np.pi*self.radius**3
            elif self.geometry == pybullet.GEOM_CYLINDER:
                self.volume = np.pi*self.radius**2*self.height
            elif self.geometry == pybullet.GEOM_CAPSULE:
                volume_sphere = 4/3*np.pi*self.radius**3
                volume_cylinder = np.pi*self.radius**2*self.height
                self.volume = volume_sphere + volume_cylinder
            self.mass = self.density*self.volume
        self.position = np.array(
            kwargs.pop("position", [0, 0, 0])
        )*self.units.meters
        self.orientation = pybullet.getQuaternionFromEuler(
            kwargs.pop("orientation", [0, 0, 0])
        )
        self.frame_position = np.array(
            kwargs.pop("frame_position", [0, 0, 0])
        )*self.units.meters
        self.frame_orientation = kwargs.pop("frame_orientation", [0, 0, 0])
        if len(self.frame_orientation) == 3:
            self.frame_orientation = pybullet.getQuaternionFromEuler(
                self.frame_orientation
            )
        self.inertial_position = kwargs.pop(
            "inertial_position",
            None
        )
        if self.inertial_position is None:
            self.inertial_position = (
                self.frame_position + (
                    self.scale*tri.load_mesh(self.filename).center_mass
                )
                if self.geometry is pybullet.GEOM_MESH
                else self.frame_position
            )
        self.inertial_position = (
            np.array(self.inertial_position)*self.units.meters
        )
        self.inertial_orientation = kwargs.pop(
            "inertial_orientation",
            self.frame_orientation
        )
        if len(self.inertial_orientation) == 3:
            self.inertial_orientation = pybullet.getQuaternionFromEuler(
                self.inertial_orientation
            )
        self.parent = kwargs.pop("parent", None)
        collision_options = kwargs.pop("collision_options", {})
        self.collision = pybullet.createCollisionShape(
            shapeType=self.geometry,
            collisionFramePosition=self.frame_position,
            collisionFrameOrientation=self.frame_orientation,
            **additional_kwargs,
            **collision_options
        )
        color = kwargs.pop("color", None)
        if "height" in additional_kwargs:
            additional_kwargs["length"] = additional_kwargs.pop("height")
        visual_options = kwargs.pop("visual_options", {})
        if visual_options:
            if color is None:
                color = [1, 1, 1, 1]
        self.visual = (
            -1
            if color is None
            else pybullet.createVisualShape(
                shapeType=self.geometry,
                visualFramePosition=self.frame_position,
                visualFrameOrientation=self.frame_orientation,
                rgbaColor=color,
                **additional_kwargs,
                **visual_options
            )
        )
        print(self.visual)

        # Joint
        self.joint_type = kwargs.pop("joint_type", pybullet.JOINT_REVOLUTE)
        self.joint_axis = kwargs.pop("joint_axis", [0, 0, 1])

        # Other
        self.update(**kwargs)
