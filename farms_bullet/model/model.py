"""Simulation model"""

import os
import numpy as np
import pybullet

import farms_pylog as pylog
from farms_data.units import SimulationUnitScaling
from .options import SpawnLoader
from ..utils.sdf import load_sdf
from ..utils.output import redirect_output


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

    def links_identities(self):
        """Joints"""
        return np.arange(-1, pybullet.getNumJoints(self._identity), dtype=int)

    def joints_identities(self):
        """Joints"""
        return np.arange(pybullet.getNumJoints(self._identity), dtype=int)

    def n_joints(self):
        """Get number of joints"""
        return pybullet.getNumJoints(self._identity)

    @staticmethod
    def get_parent_links_info(identity, base_link='base_link'):
        """Get links (parent of joint)"""
        links = {base_link: -1}
        links.update({
            info[12].decode('UTF-8'): info[16] + 1
            for info in [
                pybullet.getJointInfo(identity, j)
                for j in range(pybullet.getNumJoints(identity))
            ]
        })
        return links

    @staticmethod
    def get_joints_info(identity):
        """Get joints"""
        joints = {
            info[1].decode('UTF-8'): info[0]
            for info in [
                pybullet.getJointInfo(identity, j)
                for j in range(pybullet.getNumJoints(identity))
            ]
        }
        return joints

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
        assert os.path.isfile(sdf), '{} does not exist'.format(sdf)
        spawn_loader = kwargs.pop('spawn_loader', SpawnLoader.FARMS)
        if spawn_loader == SpawnLoader.PYBULLET:
            with redirect_output(pylog.warning):
                model = pybullet.loadSDF(sdf, **kwargs)[0]
        else:
            model = load_sdf(sdf, force_concave=True, **kwargs)[0]
        return model

    @staticmethod
    def from_urdf(urdf, **kwargs):
        """Model from SDF"""
        assert os.path.isfile(urdf), '{} does not exist'.format(urdf)
        with redirect_output(pylog.warning):
            model = pybullet.loadURDF(urdf, **kwargs)
        return model


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

    def __iter__(self):
        return iter(self._models)

    def __getitem__(self, key):
        assert key < len(self._models)
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
