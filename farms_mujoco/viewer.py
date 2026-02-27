"""Viewer"""

import numpy as np
from dm_control.mjcf.physics import Physics

from farms_core.model.data import AnimatData
from farms_core.model.options import AnimatOptions
from farms_core.model.extensions import AnimatExtension
from farms_core.experiment.options import ExperimentOptions
from farms_core.sensors.data import LinkSensorArray

from .simulation.task import ExperimentTask
from .simulation.extensions import create_cylinder


class Targets2Reach(AnimatExtension):
    """Drive control"""

    def __init__(self, config, links):
        super().__init__()
        self.config = config
        self.links: LinkSensorArray = links
        self.viewer = None
        self.cylinders = []
        self.azimuth = 0
        self.distance = 0
        self.elevation = 0
        self.cylinders = []

    @classmethod
    def from_options(
            cls,
            config: dict,
            experiment_options: ExperimentOptions,
            animat_i: int,
            animat_data: AnimatData,
            animat_options: AnimatOptions,
    ):
        """From options"""
        del animat_i, experiment_options, animat_options
        return cls(
            config=config,
            links=animat_data.sensors.links,
        )

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        """Initialize episode"""
        del physics
        self.viewer = task.viewer
        if self.viewer is not None:
            self.azimuth = self.viewer.cam.azimuth
            self.distance = self.viewer.cam.distance
            self.elevation = self.viewer.cam.elevation
            self.cylinders = [
                create_cylinder(
                    self.viewer,
                    pos=target['position'],
                    size=target['size'],
                    rgba=target['rgba1'],
                )
                for target in self.config['targets']
            ]

    def after_step(self, task: ExperimentTask, physics: Physics):
        """After step"""
        del physics
        if self.viewer:
            pos = np.array(self.links.global_com_position(
                iteration=task.iteration-1,
            ))
            for target, cylinder in zip(self.config['targets'], self.cylinders):
                if np.linalg.norm(cylinder.pos[:2] - pos[:2]) < cylinder.size[0]:
                    cylinder.rgba = target['rgba2']
