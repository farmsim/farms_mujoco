"""Animat link"""

import pybullet


class AnimatLink:
    """Animat link"""

    def __init__(self, **kwargs):
        super(AnimatLink, self).__init__()
        additional_kwargs = {}
        self.size = kwargs.pop("size", None)
        self.radius = kwargs.pop("radius", None)
        self.height = kwargs.pop("height", None)
        self.filename = kwargs.pop("filename", None)
        if self.size is not None:
            additional_kwargs["halfExtents"] = self.size
        if self.radius is not None:
            additional_kwargs["radius"] = self.radius
        if self.height is not None:
            additional_kwargs["height"] = self.height
        if self.filename is not None:
            additional_kwargs["fileName"] = self.filename
        self.geometry = kwargs.pop("geometry", pybullet.GEOM_BOX)
        self.position = kwargs.pop("position", [0, 0, 0])
        self.orientation = pybullet.getQuaternionFromEuler(
            kwargs.pop("orientation", [0, 0, 0])
        )
        self.f_position = kwargs.pop("f_position", [0, 0, 0])
        self.f_orientation = pybullet.getQuaternionFromEuler(
            kwargs.pop("f_orientation", [0, 0, 0])
        )
        self.mass = kwargs.pop("mass", 0)
        self.parent = kwargs.pop("parent", None)
        self.frame_position = kwargs.pop("frame_position", [0, 0, 0])
        self.collision = pybullet.createCollisionShape(
            self.geometry,
            collisionFramePosition=self.frame_position,
            collisionFrameOrientation=self.orientation,
            **additional_kwargs
        )
        color = kwargs.pop("color", None)
        self.visual = -1 if color is None else pybullet.createVisualShape(
            self.geometry,
            visualFramePosition=self.frame_position,
            visualFrameOrientation=self.orientation,
            rgbaColor=color,
            **additional_kwargs
        )

        # Joint
        self.joint_type = kwargs.pop("joint_type", pybullet.JOINT_REVOLUTE)
        self.joint_axis = kwargs.pop("joint_axis", [0, 0, 1])
