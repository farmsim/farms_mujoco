"""Animat link"""

import numpy as np
import trimesh as tri
import pybullet

class AnimatLink(dict):
    """Animat link"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, **kwargs):
        super(AnimatLink, self).__init__()
        additional_kwargs = {}
        self.size = kwargs.pop("size", None)
        self.radius = kwargs.pop("radius", None)
        self.height = kwargs.pop("height", None)
        self.filename = kwargs.pop("filename", None)
        self.mass = kwargs.pop("mass", None)
        self.volume = kwargs.pop("volume", None)
        self.density = kwargs.pop("density", 1000)
        if self.size is not None:
            additional_kwargs["halfExtents"] = self.size
        if self.radius is not None:
            additional_kwargs["radius"] = self.radius
        if self.height is not None:
            additional_kwargs["height"] = self.height
        if self.filename is not None:
            additional_kwargs["fileName"] = self.filename
            if self.mass is None:
                self.volume = tri.load_mesh(self.filename).volume
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
        self.f_position = kwargs.pop("f_position", self.frame_position)
        self.f_orientation = kwargs.pop("f_orientation", self.frame_orientation)
        if len(self.f_orientation) == 3:
            self.f_orientation = pybullet.getQuaternionFromEuler(
                self.f_orientation
            )
        self.parent = kwargs.pop("parent", None)
        self.collision = pybullet.createCollisionShape(
            shapeType=self.geometry,
            collisionFramePosition=self.frame_position,
            collisionFrameOrientation=self.frame_orientation,
            **additional_kwargs
        )
        color = kwargs.pop("color", None)
        self.visual = -1 if color is None else pybullet.createVisualShape(
            shapeType=self.geometry,
            visualFramePosition=self.frame_position,
            visualFrameOrientation=self.frame_orientation,
            rgbaColor=color,
            **additional_kwargs
        )

        # Joint
        self.joint_type = kwargs.pop("joint_type", pybullet.JOINT_REVOLUTE)
        self.joint_axis = kwargs.pop("joint_axis", [0, 0, 1])

        # Other
        self.update(**kwargs)
