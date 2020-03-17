"""Simulation model"""

import os
import numpy as np
import pybullet


class SimulationModel:
    """SimulationModel"""

    def __init__(self, identity=None):
        super(SimulationModel, self).__init__()
        self._identity = identity

    @property
    def identity(self):
        """Model identity"""
        return self._identity

    @staticmethod
    def spawn():
        """Spawn"""

    @staticmethod
    def step():
        """Step"""

    @staticmethod
    def log():
        """Log"""

    @staticmethod
    def save_logs():
        """Save logs"""

    @staticmethod
    def plot():
        """Plot"""

    @staticmethod
    def reset():
        """Reset"""

    @staticmethod
    def delete():
        """Delete"""

    @staticmethod
    def from_sdf(sdf, **kwargs):
        """Model from SDF"""
        return pybullet.loadSDF(sdf, **kwargs)[0]

    @staticmethod
    def from_urdf(urdf, **kwargs):
        """Model from SDF"""
        return pybullet.loadURDF(urdf, **kwargs)


class SimulationModels(SimulationModel):
    """Multiple models"""

    def __init__(self, models):
        super(SimulationModels, self).__init__()
        self.models = models

    def spawn(self):
        """Spawn"""
        for model in self.models:
            model.spawn()


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
