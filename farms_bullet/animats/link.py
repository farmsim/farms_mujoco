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
        self.radius = kwargs.pop("radius", None)
        self.height = kwargs.pop("height", None)
        self.filename = kwargs.pop("filename", None)
        self.mass = kwargs.pop("mass", None)
        self.volume = kwargs.pop("volume", None)
        self.density = kwargs.pop("density", 1000)
        self.scale = np.array(kwargs.pop("scale", [1.0, 1.0, 1.0]))
        if self.size is not None:
            additional_kwargs["halfExtents"] = np.array(
                self.size
            )*self.units.meters
        if self.radius is not None:
            additional_kwargs["radius"] = self.radius*self.units.meters
        if self.height is not None:
            additional_kwargs["height"] = self.height*self.units.meters
        if self.filename is not None:
            additional_kwargs["fileName"] = self.filename
            if self.mass is None:
                additional_kwargs["meshScale"] = self.scale*self.units.meters
                self.volume = (
                    self.scale[0]*self.scale[1]*self.scale[2]
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
        self.position = kwargs.pop("position", [0, 0, 0])
        self.orientation = pybullet.getQuaternionFromEuler(
            kwargs.pop("orientation", [0, 0, 0])
        )
        self.frame_position = kwargs.pop("frame_position", [0, 0, 0])
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
                    self.scale*tri.load_mesh(
                        self.filename
                    ).center_mass
                )
                if self.geometry is pybullet.GEOM_MESH
                else self.frame_position
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
        if collision_options:
            raise Exception("Check for scaling")
        self.collision = pybullet.createCollisionShape(
            shapeType=self.geometry,
            collisionFramePosition=np.array(
                self.frame_position
            )*self.units.meters,
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
        if visual_options:
            raise Exception("Check for scaling")
        self.visual = (
            -1
            if color is None
            else pybullet.createVisualShape(
                shapeType=self.geometry,
                visualFramePosition=np.array(
                    self.frame_position
                )*self.units.meters,
                visualFrameOrientation=self.frame_orientation,
                rgbaColor=color,
                **additional_kwargs,
                **visual_options
            )
        )

        # Joint
        self.joint_type = kwargs.pop("joint_type", pybullet.JOINT_REVOLUTE)
        self.joint_axis = kwargs.pop("joint_axis", [0, 0, 1])

        # Other
        self.update(**kwargs)
