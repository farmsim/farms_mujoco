"""Extensions"""

import os
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from dm_control.mjcf.physics import Physics

from farms_core.doc import ExtensionDoc, ChildDoc
from farms_core.options import Options
from farms_core.sensors.data import LinkSensorArray
from farms_core.simulation.extensions import TaskExtension
from farms_core.experiment.options import ExperimentOptions
from farms_core.units import SimulationUnitScaling

from .task import ExperimentTask
from .mjcf import mjcf2str


def create_primitive(scn, primitive, **kwargs):
    """Create primitive geom on a MuJoCo scene.

    Adds a geom of the given ``primitive`` type to ``scn`` and advances
    ``scn.ngeom``.  Returns ``None`` if the scene is full.
    """
    if scn.ngeom >= scn.maxgeom:
        return None
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom=geom,
        type=primitive,
        size=kwargs.pop('size', [1.0, 1.0, 1.0]),
        pos=kwargs.pop('pos', [0.0, 0.0, 0.0]),  # Pos
        mat=kwargs.pop('mat', np.eye(3).ravel()),  # Matrix
        rgba=kwargs.pop('rgba', [1.0, 1.0, 1.0, 1.0]),  # RGBA
        **kwargs,
    )
    scn.ngeom += 1
    return geom


def create_sphere(scn, **kwargs):
    """Create sphere geom on a MuJoCo scene.

    Works with either ``viewer.user_scn`` (pass ``viewer``) or a raw
    ``MjvScene`` (pass the scene directly).
    """
    return create_primitive(
        getattr(scn, 'user_scn', scn),
        mujoco.mjtGeom.mjGEOM_SPHERE,
        **kwargs,
    )


def create_cylinder(scn, **kwargs):
    """Create cylinder geom on a MuJoCo scene.

    Works with either ``viewer.user_scn`` (pass ``viewer``) or a raw
    ``MjvScene`` (pass the scene directly).
    """
    return create_primitive(
        getattr(scn, 'user_scn', scn),
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        **kwargs,
    )


def create_line(scn, begin, end, **kwargs):
    """Create line geom connecting two points on a MuJoCo scene.

    Works with either ``viewer.user_scn`` (pass ``viewer``) or a raw
    ``MjvScene`` (pass the scene directly).  Returns ``None`` if the
    scene is full.
    """
    scn = getattr(scn, 'user_scn', scn)
    if scn.ngeom >= scn.maxgeom:
        return None
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom=geom,
        type=mujoco.mjtGeom.mjGEOM_LINE,
        size=[1.0, 1.0, 1.0],  # Size
        pos=begin,  # Pos
        mat=np.eye(3).ravel(),  # Matrix
        rgba=kwargs.pop('rgba', [1.0, 0.3, 0.0, 0.7]),  # RGBA
    )
    mujoco.mjv_connector(
        geom=geom,
        type=mujoco.mjtGeom.mjGEOM_LINE,
        width=kwargs.pop('width', 5),  # Width
        from_=begin,
        to=end,
    )
    scn.ngeom += 1
    return geom


def create_arrow(scn, **kwargs):
    """Create arrow geom on a MuJoCo scene.

    Works with either ``viewer.user_scn`` (pass ``viewer``) or a raw
    ``MjvScene`` (pass the scene directly).
    """
    return create_primitive(
        getattr(scn, 'user_scn', scn),
        mujoco.mjtGeom.mjGEOM_ARROW,
        **kwargs,
    )


class AnimatViewerExtension(TaskExtension):
    """Base class for extensions that operate on a specific animat.

    Provides shared helpers for binding to an animat's link sensors and
    querying its global centre-of-mass position.
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs.pop('substep', False))
        self.links: LinkSensorArray | None = None
        self.units: SimulationUnitScaling | None = None
        self.animat_id = kwargs.pop('animat_id', 0)
        self.show_on_camera = kwargs.pop('show_on_camera', True)

    @classmethod
    def from_options(cls, config: dict, experiment_options: ExperimentOptions):
        """Not all extensions support config-based construction."""
        raise NotImplementedError(
            f"{cls.__name__} does not support from_options"
        )

    def bind_links(self, task: ExperimentTask):
        """Bind to the animat's link sensors."""
        self.links = task.data.animats[self.animat_id].sensors.links

    def com_position(self, iteration: int) -> np.ndarray:
        """Return the global CoM position at the given iteration."""
        return np.array(
            self.links.global_com_position(iteration=iteration)
        )

    def com_radius(self) -> float | None:
        """Compute sphere radius from total body mass.

        Returns ``None`` if mass data is unavailable.
        """
        mass = np.sum(self.links.masses)
        if mass is not None:
            return 0.2*((3*mass/1000)/np.pi)**(1/3)
        return None


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


class CameraFollower(AnimatViewerExtension):
    """Camera follower viewer"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.azimuth = kwargs.pop('azimuth', 0)
        self.distance = kwargs.pop('distance', 1)
        self.elevation = kwargs.pop('elevation', 0)
        self.angular_velocity = kwargs.pop('angular_velocity', 0)  # [deg/s]
        self.viewer = kwargs.pop('viewer', None)
        self.camera = kwargs.pop('camera', None)
        if self.units is None:
            self.units = kwargs.pop('units', SimulationUnitScaling())
        else:
            kwargs.pop('units', None)
        self.last_step = 0

    @classmethod
    def from_options(cls, config: dict, experiment_options: ExperimentOptions):
        """From options"""
        return cls(**CameraFollowerOptions(
            # timestep=experiment_options.simulation.phxsics.timestep,
            units=experiment_options.simulation.units,
            **config,
        ))

    def get_camera(self):
        """Return the camera object to update.

        Returns ``self.camera`` if set (viewport use), otherwise
        ``self.viewer.cam`` (standalone viewer use).
        """
        if self.camera is not None:
            return self.camera
        return self.viewer.cam if self.viewer else None

    def update_camera_lookat(self, cam, task, physics, units):
        """Smoothly interpolate camera lookat toward the animat CoM.

        Core math shared between ``CameraFollower`` and
        ``ViewportCameraFollower``.  ``cam`` is the MjvCamera to update;
        ``units`` is the unit scaling to use.
        """
        now = physics.time()/units.seconds
        time_diff, self.last_step = now - self.last_step, now
        cam.azimuth += self.angular_velocity*time_diff
        motion_filter = min(1, 10*physics.timestep()/units.seconds)
        cam.lookat = motion_filter*self.com_position(
            iteration=task.iteration - 1,
        )*units.meters + (1 - motion_filter)*cam.lookat

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        """Initialise episode"""
        del physics
        self.units = task.units
        self.viewer = task.viewer
        self.last_step = 0
        self.bind_links(task)
        cam = self.get_camera()
        if cam is not None:
            cam.azimuth = self.azimuth
            cam.distance = self.distance*self.units.meters
            cam.elevation = self.elevation

    def after_step(self, task: ExperimentTask, physics: Physics):
        """After step"""
        cam = self.get_camera()
        if cam is not None and self.links is not None:
            self.update_camera_lookat(cam, task, physics, self.units)


class CoMViewer(AnimatViewerExtension):
    """CoM viewer"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sphere = None
        self.com: np.ndarray | None = None
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

    def render(self, scene, pos):
        """Add a CoM sphere geom to the scene at the given position.

        ``pos`` is in SI units (meters); converted to MuJoCo units here.
        Works with either a ``viewer`` (accessing ``viewer.user_scn``)
        or a raw ``MjvScene``.
        """
        return create_sphere(
            scene,
            size=[s*self.units.meters for s in self.size],
            pos=[p*self.units.meters for p in pos],
            rgba=self.rgba,
        )

    def render_scene(self, scene):
        """Render the CoM sphere onto a scene.

        Called after ``mjv_updateScene`` clears the scene, before
        ``mjr_render``.
        """
        if self.com is not None:
            self.render(scene, self.com)

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        """Initialise episode"""
        del physics
        self.units = task.units
        self.viewer = task.viewer
        self.bind_links(task)
        radius = self.com_radius()
        if radius is not None:
            self.size = [radius, 0.0, 0.0]
        self.com = self.com_position(0)
        if self.viewer:
            self.sphere = self.render(self.viewer.user_scn, self.com)

    def after_step(self, task: ExperimentTask, physics: Physics):
        del physics
        if self.links is None:
            return
        self.com = self.com_position(task.iteration - 1)
        if self.sphere:
            self.sphere.pos = self.com*self.units.meters


class TrailCoMViewer(AnimatViewerExtension):
    """CoM trail viewer"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pos_old = None
        self.pos_new = None
        self.segments: list[tuple[np.ndarray, np.ndarray]] = []
        self.width = kwargs.pop('width', 5)
        self.rgba = kwargs.pop('rgba', [1.0, 0.3, 0.0, 0.7])
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

    def render(self, scene, begin, end):
        """Add a trail line geom to the scene connecting two points.

        ``begin`` and ``end`` are in SI units (meters); converted to
        MuJoCo units here.  Works with either a ``viewer`` (accessing
        ``viewer.user_scn``) or a raw ``MjvScene``.
        """
        return create_line(
            scene,
            begin*self.units.meters,
            end*self.units.meters,
            width=self.width,
            rgba=self.rgba,
        )

    def render_scene(self, scene):
        """Render all trail segments onto a scene.

        Called after ``mjv_updateScene`` clears the scene, before
        ``mjr_render``.
        """
        for begin, end in self.segments:
            self.render(scene, begin, end)

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        """Initialise episode"""
        del physics
        self.units = task.units
        self.viewer = task.viewer
        self.bind_links(task)
        self.segments = []
        self.pos_new = self.pos_old = self.com_position(0)

    def after_step(self, task: ExperimentTask, physics: Physics):
        del physics
        if self.links is None:
            return
        iteration = task.iteration - 1
        if not iteration % self.spacing:
            self.pos_new = self.com_position(iteration)
            if self.pos_old is not None:
                self.segments.append((self.pos_old.copy(), self.pos_new.copy()))
                if self.viewer:
                    self.render(self.viewer.user_scn, self.pos_old, self.pos_new)
            self.pos_old = self.pos_new


class TrailLinkViewer(AnimatViewerExtension):
    """Link trail viewer"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pos_old = None
        self.pos_new = None
        self.width = kwargs.pop('width', 5)
        self.rgba = kwargs.pop('rgba', [1.0, 0.3, 0.0, 0.7])
        self.link_name: str = kwargs.pop('link', '')
        self.link_id: int = kwargs.pop('link_id', None)
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
        self.units = task.units
        self.viewer = task.viewer
        self.bind_links(task)
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
                self.viewer.user_scn,
                self.pos_old*self.units.meters,
                self.pos_new*self.units.meters,
                width=5,
                rgba=self.rgba,
            )
            self.pos_old = self.pos_new


class ArrowViewer(AnimatViewerExtension):
    """CoM viewer"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.arrow = None
        self.size = kwargs.pop('size', [0.03, 0.03, 0.3])
        self.rgba = kwargs.pop('rgba', [1.0, 1.0, 1.0, 0.3])
        self.viewer = kwargs.pop('viewer', None)
        self.offset = kwargs.pop('offset', None)
        self.orientation = 0

    @classmethod
    def from_options(cls, config: dict, experiment_options: ExperimentOptions):
        """From options"""
        del experiment_options
        return cls(
            animat_id=config.get('animat_id', 0),
            size=config.get('size', None),
            offset=config.get('offset', None),
            rgba=config['rgba'],
        )

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        """Initialise episode"""
        self.units = task.units
        self.viewer = task.viewer
        self.bind_links(task)
        if self.viewer:
            if self.size is None:
                radius = self.com_radius()
                if radius is not None:
                    self.size = [0.5*radius, 0.5*radius, 20*radius]
                    self.offset = 5*radius
                else:
                    self.size = [0.03, 0.03, 0.3]
                    self.offset = 0.5
            size_mujoco = [s*self.units.meters for s in self.size]
            self.arrow = create_arrow(
                self.viewer.user_scn,
                size=size_mujoco,
                rgba=self.rgba,
            )
            self.arrow.size = size_mujoco

    def after_step(self, task: ExperimentTask, physics: Physics):
        time = physics.time()/task.units.seconds
        if self.arrow and self.links is not None:
            self.arrow.pos = (
                self.com_position(iteration=task.iteration-1)
                + np.array([0, 0, self.offset])
            )*self.units.meters
            self.arrow.mat = Rotation.from_euler(
                seq='xyz',
                angles=[0.5*np.pi, 0, 0.2*np.pi*time],
            ).as_matrix()
