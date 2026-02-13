"""Extensions"""

import os
from dataclasses import dataclass

import mujoco
import numpy as np
from dm_control.mjcf.physics import Physics

from farms_core.doc import ExtensionDoc, ChildDoc
from farms_core.options import Options
from farms_core.sensors.data import LinkSensorArray
from farms_core.simulation.extensions import TaskExtension
from farms_core.experiment.options import ExperimentOptions
from farms_core.units import SimulationUnitScaling

from .task import ExperimentTask
from .mjcf import mjcf2str


def create_primitive(viewer, primitive, **kwargs):
    """Create primitive"""
    scn = viewer.user_scn
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        primitive,
        kwargs.pop('size', [1.0, 1.0, 1.0]),
        kwargs.pop('pos', [0.0, 0.0, 0.0]),  # Pos
        np.eye(3).ravel(),  # Matrix
        kwargs.pop('rgba', [1.0, 1.0, 1.0, 1.0]),  # RGBA
    )
    scn.ngeom += 1
    return geom


def create_sphere(viewer, **kwargs):
    """Create sphere"""
    return create_primitive(
        viewer,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        **kwargs,
    )


def create_cylinder(viewer, **kwargs):
    """Create cylinder"""
    return create_primitive(
        viewer,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        **kwargs,
    )


def create_line(viewer, begin, end, **kwargs):
    """Create sphere"""
    scn = viewer.user_scn
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        [1.0, 1.0, 1.0],  # Size
        begin,  # Pos
        np.eye(3).ravel(),  # Matrix
        kwargs.pop('rgba', [1.0, 0.3, 0.0, 0.7]),  # RGBA
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        kwargs.pop('width', 5),  # Width
        begin,
        end,
    )
    scn.ngeom += 1
    return geom


class MjcfSaver(TaskExtension):
    """CoM viewer"""

    def __init__(self, path):
        super().__init__()
        self.path = path

    @classmethod
    def from_options(cls, config: dict, experiment_options: ExperimentOptions):
        """From options"""
        del experiment_options
        return cls(
            path=config.get('path', 'simulation_mjcf.xml'),
        )

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        """Initialise episode"""
        del physics
        dir_path = os.path.dirname(self.path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        mjcf_xml_str = mjcf2str(mjcf_model=task.mjcf)
        with open(self.path, 'w+', encoding='utf-8') as xml_file:
            xml_file.write(mjcf_xml_str)


@dataclass
class CameraFollowerOptions(Options):
    """Camera follower viewer options"""

    @classmethod
    def doc(cls):
        """Doc"""
        return ExtensionDoc(
            name="Camera follower options",
            description="Describes the camera options.",
            class_type=cls,
            children=[
                ChildDoc(
                    name="distance",
                    class_type=float,
                    description="Camera zoom.",
                ),
                ChildDoc(
                    name="free_camera",
                    class_type=bool,
                    description=(
                        "Whether the camera should be free moving instead of"
                        " following the animat."
                    ),
                ),
                ChildDoc(
                    name="top_camera",
                    class_type=bool,
                    description=(
                        "Whether the camera should look at the animat from"
                        " above."
                    ),
                ),
                ChildDoc(
                    name="rotating_camera",
                    class_type=bool,
                    description=(
                        "Whether the camera should turn around the model."
                    ),
                ),
            ],
        )

    def __init__(self, **kwargs):
        super().__init__()
        self.animat_id = kwargs.pop('animat_id', 0)
        self.azimuth = kwargs.pop('azimuth', 0)
        self.distance = kwargs.pop('distance', 1)
        self.elevation = kwargs.pop('elevation', 0)
        self.angular_velocity = kwargs.pop('angular_velocity', 0)  # [deg/s]
        self.units = kwargs.pop('units', SimulationUnitScaling())
        assert not kwargs, kwargs


class CameraFollower(TaskExtension):
    """Camera follower viewer"""

    def __init__(self, **kwargs):
        super().__init__()
        self.links: LinkSensorArray | None = None
        self.animat_id = kwargs.pop('animat_id', 0)
        self.azimuth = kwargs.pop('azimuth', 0)
        self.distance = kwargs.pop('distance', 1)
        self.elevation = kwargs.pop('elevation', 0)
        self.angular_velocity = kwargs.pop('angular_velocity', 0)  # [deg/s]
        self.viewer = kwargs.pop('viewer', None)
        self.units = kwargs.pop('units', SimulationUnitScaling())
        self.last_step = 0

    @classmethod
    def from_options(cls, config: dict, experiment_options: ExperimentOptions):
        """From options"""
        return cls(**CameraFollowerOptions(
            # timestep=experiment_options.simulation.phxsics.timestep,
            units=experiment_options.simulation.units,
            **config,
        ))

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        """Initialise episode"""
        del physics
        self.viewer = task.viewer
        if self.viewer:
            self.viewer.cam.azimuth = self.azimuth
            self.viewer.cam.distance = self.distance*self.units.meters
            self.viewer.cam.elevation = self.elevation
            self.links = task.data.animats[self.animat_id].sensors.links

    def after_step(self, task: ExperimentTask, physics: Physics):
        """After step"""
        if self.viewer and self.links is not None:
            now = physics.time()/task.units.seconds
            time_diff, self.last_step = now - self.last_step, now
            self.viewer.cam.azimuth += self.angular_velocity*time_diff
            self.motion_filter = min(1, 10*physics.timestep()/task.units.seconds)
            self.viewer.cam.lookat = self.motion_filter*np.array(
                self.links.global_com_position(iteration=task.iteration-1)
            )*self.units.meters + (1-self.motion_filter)*self.viewer.cam.lookat


class CoMViewer(TaskExtension):
    """CoM viewer"""

    def __init__(self, **kwargs):
        super().__init__()
        self.sphere = None
        self.animat_id = kwargs.pop('animat_id', 0)
        self.links: LinkSensorArray | None = None
        self.size = kwargs.pop('size', [0.01, 0.0, 0.0])
        self.rgba = kwargs.pop('rgba', [1.0, 1.0, 1.0, 0.3])
        self.viewer = kwargs.pop('viewer', None)

    @classmethod
    def from_options(cls, config: dict, experiment_options: ExperimentOptions):
        """From options"""
        del experiment_options
        return cls(
            animat_id=config.get('animat_id', 0),
            size=config['size'],
            rgba=config['rgba'],
        )

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        """Initialise episode"""
        self.viewer = task.viewer
        index = self.animat_id
        self.links: LinkSensorArray = task.data.animats[index].sensors.links
        mass = np.sum(self.links.masses)
        if self.viewer:
            self.sphere = create_sphere(
                self.viewer,
                size=self.size,
                rgba=self.rgba,
            )
            if mass is not None:
                radius = 0.2*((3*mass/1000)/np.pi)**(1/3)
                self.sphere.size[:] = radius

    def after_step(self, task: ExperimentTask, physics: Physics):
        del physics
        if self.sphere and self.links is not None:
            self.sphere.pos = np.array(self.links.global_com_position(
                iteration=task.iteration-1,
            ))


class TrailCoMViewer(TaskExtension):
    """CoM trail viewer"""

    def __init__(self, **kwargs):
        super().__init__()
        self.pos_old = None
        self.pos_new = None
        self.width = kwargs.pop('width', 5)
        self.rgba = kwargs.pop('rgba', [1.0, 0.3, 0.0, 0.7])
        self.animat_id = kwargs.pop('animat_id', 0)
        self.links: LinkSensorArray | None = None
        self.viewer = kwargs.pop('viewer', None)
        self.spacing = kwargs.pop('spacing', 10)

    @classmethod
    def from_options(cls, config: dict, experiment_options: ExperimentOptions):
        """From options"""
        del experiment_options
        return cls(
            animat_id=config.get('animat_id', 0),
            size=config['width'],
            rgba=config['rgba'],
        )

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        """Initialise episode"""
        self.viewer = task.viewer
        index = self.animat_id
        self.links: LinkSensorArray = task.data.animats[index].sensors.links
        self.pos_new = self.pos_old = self.links.global_com_position(0)

    def after_step(self, task: ExperimentTask, physics: Physics):
        del physics
        iteration = task.iteration-1
        if (
                self.viewer
                and self.links is not None
                and not iteration % self.spacing
        ):
            self.pos_new = self.links.global_com_position(iteration)
            create_line(
                self.viewer,
                self.pos_old,
                self.pos_new,
                width=self.width,
                rgba=self.rgba,
            )
            self.pos_old = self.pos_new


class TrailLinkViewer(TaskExtension):
    """Link trail viewer"""

    def __init__(self, **kwargs):
        super().__init__()
        self.pos_old = None
        self.pos_new = None
        self.width = kwargs.pop('width', 5)
        self.rgba = kwargs.pop('rgba', [1.0, 0.3, 0.0, 0.7])
        self.link_name: str = kwargs.pop('link', '')
        self.animat_id = kwargs.pop('animat_id', 0)
        self.link_id: int = kwargs.pop('link_id', None)
        self.links: LinkSensorArray | None = None
        self.viewer = kwargs.pop('viewer', None)
        self.spacing = kwargs.pop('spacing', 10)

    @classmethod
    def from_options(cls, config: dict, experiment_options: ExperimentOptions):
        """From options"""
        del experiment_options
        return cls(
            animat_id=config.get('animat_id', 0),
            size=config['width'],
            rgba=config['rgba'],
            link=config['link'],
        )

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        """Initialise episode"""
        del physics
        self.viewer = task.viewer
        index = self.animat_id
        self.links: LinkSensorArray = task.data.animats[index].sensors.links
        assert self.link_name in self.links.names, (
            f"{self.link_name=} not in {self.links.names=}"
        )
        self.link_id = self.links.names.index(self.link_name)
        self.pos_new = self.pos_old = self.links.com_position(
            iteration=0,
            link_i=self.link_id,
        )

    def after_step(self, task: ExperimentTask, physics: Physics):
        del physics
        iteration = task.iteration-1
        if (
                self.viewer
                and self.links is not None
                and not iteration % self.spacing
        ):
            self.pos_new = self.links.com_position(
                iteration=iteration,
                link_i=self.link_id,
            )
            create_line(
                self.viewer,
                self.pos_old,
                self.pos_new,
                width=5,
                rgba=self.rgba,
            )
            self.pos_old = self.pos_new
