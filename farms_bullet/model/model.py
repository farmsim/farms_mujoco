"""Simulation model"""

import os
import numpy as np
import pybullet
import farms_pylog as pylog


class SimulationModel:
    """SimulationModel"""

    def __init__(self, identity=None):
        super(SimulationModel, self).__init__()
        self._identity = identity
        self.joint_list = None
        self.controller = None

    def identity(self):
        """Model identity"""
        return self._identity

    def spawn(self):
        """Spawn"""

    def step(self):
        """Step"""

    def log(self):
        """Log"""

    def save_logs(self):
        """Save logs"""

    def plot(self):
        """Plot"""

    def reset(self):
        """Reset"""

    def delete(self):
        """Delete"""

    @staticmethod
    def from_sdf(sdf, **kwargs):
        """Model from SDF"""
        return pybullet.loadSDF(sdf, **kwargs)[0]

    @staticmethod
    def from_urdf(urdf, **kwargs):
        """Model from SDF"""
        return pybullet.loadURDF(urdf, **kwargs)


class GroundModel(SimulationModel):
    """DescriptionFormatModel"""

    def __init__(self, position=None, orientation=None):
        super(GroundModel, self).__init__()
        self.position = position
        self.orientation = orientation
        self.plane = None

    def spawn(self):
        """Spawn"""
        self.plane = pybullet.createCollisionShape(pybullet.GEOM_PLANE)
        options = {}
        if self.position is not None:
            options['basePosition'] = self.position
        if self.orientation is not None:
            options['baseOrientation'] = self.orientation
        self._identity = pybullet.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=self.plane,
            **options,
        )


class DescriptionFormatModel(SimulationModel):
    """DescriptionFormatModel"""

    def __init__(
            self, path,
            load_options=None,
            spawn_options=None,
            visual_options=None
    ):
        super(DescriptionFormatModel, self).__init__()
        self.path = path
        self.load_options = (
            load_options
            if load_options is not None
            else {}
        )
        self.spawn_options = (
            spawn_options
            if spawn_options is not None
            else {}
        )
        self.visual_options = (
            visual_options
            if visual_options is not None
            else {}
        )

    def spawn(self):
        """Spawn"""
        extension = os.path.splitext(self.path)[1]
        if extension == '.sdf':
            self._identity = self.from_sdf(self.path, **self.load_options)
        elif extension == '.urdf':
            self._identity = self.from_urdf(self.path, **self.load_options)
        else:
            raise Exception('Unknown description format extension .{}'.format(
                extension
            ))

        # Spawn options
        if self.spawn_options:
            pos = pybullet.getBasePositionAndOrientation(
                bodyUniqueId=self._identity
            )[0]
            pos_obj = self.spawn_options.pop('posObj')
            orn_obj = self.spawn_options.pop('ornObj')
            pybullet.resetBasePositionAndOrientation(
                bodyUniqueId=self._identity,
                posObj=np.array(pos) + np.array(pos_obj),
                ornObj=np.array(orn_obj),
            )

        # Visual options
        if self.visual_options:
            path = self.visual_options.pop('path')
            texture = pybullet.loadTexture(
                os.path.join(os.path.dirname(self.path), path)
            )
            rgba_color = self.visual_options.pop('rgbaColor')
            specular_color = self.visual_options.pop('specularColor')
            for info in pybullet.getVisualShapeData(self._identity):
                for i in range(pybullet.getNumJoints(self._identity)+1):
                    pybullet.changeVisualShape(
                        objectUniqueId=info[0],
                        linkIndex=info[1],
                        shapeIndex=-1,
                        textureUniqueId=texture,
                        rgbaColor=rgba_color,
                        specularColor=specular_color,
                        **self.visual_options,
                    )


class SimulationModels(SimulationModel):
    """Simulation models"""

    def __init__(self, models):
        super(SimulationModels, self).__init__()
        self._models = models

    def __getitem__(self, key):
        return self._models[key]

    def spawn(self):
        """Spawn"""
        for model in self:
            model.spawn()

    def step(self):
        """Step"""
        for model in self:
            model.step()

    def log(self):
        """Log"""
        for model in self:
            model.log()
