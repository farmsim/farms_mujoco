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
    """MJCF model saver"""

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
                    name="animat_id",
                    class_type=int,
                    description="Index of the animat to follow.",
                ),
                ChildDoc(
                    name="distance",
                    class_type=float,
                    description="Camera distance [m].",
                ),
                ChildDoc(
                    name="azimuth",
                    class_type=float,
                    description="Camera azimuth [deg].",
                ),
                ChildDoc(
                    name="elevation",
                    class_type=float,
                    description="Camera elevation [deg].",
                ),
                ChildDoc(
                    name="angular_velocity",
                    class_type=float,
                    description=(
                        "Camera azimuth rotation speed [deg/s]."
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
                width=self.width,
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
        self.pos: np.ndarray | None = None
        self.mat: np.ndarray | None = None

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

    def render(self, scene, pos, mat):
        """Add an arrow geom to the scene at the given pose.

        ``pos`` and ``mat`` are in SI units (meters / rotation matrix);
        converted to MuJoCo units here.  Works with either a ``viewer``
        (accessing ``viewer.user_scn``) or a raw ``MjvScene``.
        """
        size_mujoco = [s*self.units.meters for s in self.size]
        return create_arrow(
            scene,
            size=size_mujoco,
            pos=[p*self.units.meters for p in pos],
            mat=mat.ravel(),
            rgba=self.rgba,
        )

    def render_scene(self, scene):
        """Render the arrow onto a scene.

        Called after ``mjv_updateScene`` clears the scene, before
        ``mjr_render``.
        """
        if self.pos is not None and self.mat is not None:
            self.render(scene, self.pos, self.mat)

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        """Initialise episode"""
        self.units = task.units
        self.viewer = task.viewer
        self.bind_links(task)
        if self.size is None:
            radius = self.com_radius()
            if radius is not None:
                self.size = [0.5*radius, 0.5*radius, 20*radius]
                self.offset = 5*radius
            else:
                self.size = [0.03, 0.03, 0.3]
                self.offset = 0.5
        self.pos = self.com_position(0) + np.array([0, 0, self.offset])
        self.mat = Rotation.from_euler(
            seq='xyz',
            angles=[0, 0.5*np.pi, 0],
        ).as_matrix()
        if self.viewer:
            size_mujoco = [s*self.units.meters for s in self.size]
            self.arrow = create_arrow(
                self.viewer.user_scn,
                size=size_mujoco,
                rgba=self.rgba,
            )
            self.arrow.size = size_mujoco

    def after_step(self, task: ExperimentTask, physics: Physics):
        time = physics.time()/task.units.seconds
        if self.links is not None:
            self.pos = (
                self.com_position(iteration=task.iteration-1)
                + np.array([0, 0, self.offset])
            )
            self.mat = Rotation.from_euler(
                seq='xyz',
                angles=[0, 0.5*np.pi, 0.2*np.pi*time],
            ).as_matrix()
            if self.arrow:
                self.arrow.pos = self.pos*self.units.meters
                self.arrow.mat = self.mat


@dataclass
class SnakeGameOptions(Options):
    """Snake-game options"""

    @classmethod
    def doc(cls):
        """Doc"""
        return ExtensionDoc(
            name="Snake game options",
            description=(
                "Describes the options for the snake-like foraging game."
                " Good-food spheres appear at random locations; when the"
                " animat approaches one it is consumed, the score"
                " increases, energy recharges, and a new food spawns."
                " Bad-food (poison) spheres deplete energy and score."
                " Food expires after a time limit.  Energy drains"
                " continuously; when it reaches zero the game resets."
            ),
            class_type=cls,
            children=[
                ChildDoc(
                    name="animat_id",
                    class_type=int,
                    description="Index of the animat playing the game.",
                ),
                ChildDoc(
                    name="radius",
                    class_type=float,
                    description="Radius of food spheres [m].",
                ),
                ChildDoc(
                    name="catch_distance",
                    class_type=float,
                    description=(
                        "Distance [m] at which the animat is considered"
                        " to have reached a food item."
                    ),
                ),
                ChildDoc(
                    name="spawn_radius",
                    class_type=float,
                    description=(
                        "Max radius [m] of the disc within which food"
                        " spheres spawn (around the animat)."
                    ),
                ),
                ChildDoc(
                    name="min_spawn_radius",
                    class_type=float,
                    description=(
                        "Min radius [m] from the animat at which food"
                        " can spawn (spawn-area exclusion)."
                    ),
                ),
                ChildDoc(
                    name="spawn_height",
                    class_type=float,
                    description=(
                        "Fixed height [m] at which food spheres spawn."
                    ),
                ),
                ChildDoc(
                    name="n_food",
                    class_type=int,
                    description=(
                        "Number of good-food spheres active at once."
                    ),
                ),
                ChildDoc(
                    name="n_bad_food",
                    class_type=int,
                    description=(
                        "Number of bad-food (poison) spheres active at"
                        " once."
                    ),
                ),
                ChildDoc(
                    name="food_rgba",
                    class_type="list[float]",
                    description="RGBA colour of good-food spheres.",
                ),
                ChildDoc(
                    name="bad_food_rgba",
                    class_type="list[float]",
                    description="RGBA colour of bad-food spheres.",
                ),
                ChildDoc(
                    name="food_lifetime",
                    class_type=float,
                    description=(
                        "Minimum lifespan of each food item [s]."
                        " The food shrinks over its lifetime and"
                        " disappears when it expires, respawning"
                        " elsewhere.  Each item gets a random"
                        " lifetime between ``food_lifetime`` and"
                        " ``food_lifetime_max``.  Zero or negative"
                        " means food never expires."
                    ),
                ),
                ChildDoc(
                    name="food_lifetime_max",
                    class_type=float,
                    description=(
                        "Maximum lifespan of each food item [s]."
                        " Each item gets a random lifetime between"
                        " ``food_lifetime`` and ``food_lifetime_max``."
                        " If equal to ``food_lifetime``, the lifetime"
                        " is fixed."
                    ),
                ),
                ChildDoc(
                    name="energy_init",
                    class_type=float,
                    description="Initial energy level.",
                ),
                ChildDoc(
                    name="energy_max",
                    class_type=float,
                    description=(
                        "Maximum energy level (cap on recharge)."
                    ),
                ),
                ChildDoc(
                    name="energy_drain",
                    class_type=float,
                    description=(
                        "Energy drained per simulated second"
                        " [energy/s].  Default gives ~20 s to grab"
                        " food before energy runs out."
                    ),
                ),
                ChildDoc(
                    name="energy_recharge",
                    class_type=float,
                    description=(
                        "Energy restored per good-food consumed."
                    ),
                ),
                ChildDoc(
                    name="energy_bad",
                    class_type=float,
                    description="Energy lost per bad-food touched.",
                ),
                ChildDoc(
                    name="show_on_camera",
                    class_type=bool,
                    description=(
                        "Whether food spheres should be rendered on"
                        " the camera recording."
                    ),
                ),
                ChildDoc(
                    name="max_score",
                    class_type=int,
                    description=(
                        "Score at which the game is won.  Zero or"
                        " negative means endless mode."
                    ),
                ),
                ChildDoc(
                    name="display_dt",
                    class_type=float,
                    description=(
                        "Minimum interval [s] between viewer text"
                        " overlay updates.  Throttling avoids"
                        " contention on the MuJoCo render lock."
                    ),
                ),
            ],
        )

    def __init__(self, **kwargs):
        super().__init__()
        self.animat_id = kwargs.pop('animat_id', 0)
        self.radius = kwargs.pop('radius', 0.05)
        self.catch_distance = kwargs.pop('catch_distance', 0.2)
        self.spawn_radius = kwargs.pop('spawn_radius', 2.0)
        self.min_spawn_radius = kwargs.pop('min_spawn_radius', 0.3)
        self.spawn_height = kwargs.pop('spawn_height', 0.5)
        self.n_food = kwargs.pop('n_food', 1)
        self.n_bad_food = kwargs.pop('n_bad_food', 0)
        self.food_rgba = kwargs.pop('food_rgba', [0.2, 0.8, 0.2, 0.8])
        self.bad_food_rgba = kwargs.pop(
            'bad_food_rgba', [0.8, 0.2, 0.2, 0.8],
        )
        self.food_lifetime = kwargs.pop('food_lifetime', 10.0)
        self.food_lifetime_max = kwargs.pop('food_lifetime_max', 20.0)
        self.energy_init = kwargs.pop('energy_init', 100.0)
        self.energy_max = kwargs.pop('energy_max', 100.0)
        self.energy_drain = kwargs.pop('energy_drain', 5.0)
        self.energy_recharge = kwargs.pop('energy_recharge', 20.0)
        self.energy_bad = kwargs.pop('energy_bad', 30.0)
        self.show_on_camera = kwargs.pop('show_on_camera', True)
        self.max_score = kwargs.pop('max_score', 0)
        self.display_dt = kwargs.pop('display_dt', 0.05)
        assert not kwargs, kwargs


@dataclass
class _FoodItem:
    """A single food sphere in the snake game."""
    pos: np.ndarray
    is_bad: bool
    spawn_time: float
    rgba: list[float]
    lifetime: float = 0.0  # per-item lifetime [s]; 0 means never expires
    geom: object = None  # viewer.user_scn geom reference (or None)


class SnakeGame(AnimatViewerExtension):
    """Snake-like foraging game with energy and difficulty scaling.

    Good-food spheres spawn at random locations within ``spawn_radius``
    (but not closer than ``min_spawn_radius``) of the animat's centre
    of mass.  When the animat's CoM comes within ``catch_distance`` of a
    food item, it is consumed: the score increases, energy is
    recharged, and a new food item spawns elsewhere.

    Bad-food (poison) spheres deplete energy and score on contact.

    Each food item has a lifespan; if not consumed in time it expires
    and respawns elsewhere.  The lifespan shortens as the score grows.

    Energy drains continuously; when it reaches zero the game resets
    (score back to zero, energy restored, all food respawned).

    The score and energy are displayed as an overlay in the MuJoCo
    viewer (top-left corner) and are also available programmatically
    via ``self.score`` and ``self.energy``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.radius = kwargs.pop('radius', 0.05)
        self.catch_distance = kwargs.pop('catch_distance', 0.2)
        self.spawn_radius = kwargs.pop('spawn_radius', 2.0)
        self.min_spawn_radius = kwargs.pop('min_spawn_radius', 0.3)
        self.spawn_height = kwargs.pop('spawn_height', 0.5)
        self.n_food = kwargs.pop('n_food', 1)
        self.n_bad_food = kwargs.pop('n_bad_food', 0)
        self.food_rgba = kwargs.pop('food_rgba', [0.2, 0.8, 0.2, 0.8])
        self.bad_food_rgba = kwargs.pop(
            'bad_food_rgba', [0.8, 0.2, 0.2, 0.8],
        )
        self.food_lifetime = kwargs.pop('food_lifetime', 10.0)
        self.food_lifetime_max = kwargs.pop('food_lifetime_max', 20.0)
        self.energy_init = kwargs.pop('energy_init', 100.0)
        self.energy_max = kwargs.pop('energy_max', 100.0)
        self.energy_drain = kwargs.pop('energy_drain', 5.0)
        self.energy_recharge = kwargs.pop('energy_recharge', 20.0)
        self.energy_bad = kwargs.pop('energy_bad', 30.0)
        self.max_score = kwargs.pop('max_score', 0)
        self.viewer = kwargs.pop('viewer', None)
        self.score = 0
        self.energy = self.energy_init
        self.foods: list[_FoodItem] = []
        self._rng = np.random.default_rng()
        # Throttle viewer text overlay updates to avoid contention on
        # the MuJoCo render lock (set_texts is expensive when called
        # every physics sub-step).
        self._display_dt = kwargs.pop('display_dt', 0.05)
        self._last_display_time = -1.0

    @classmethod
    def from_options(
        cls, config: dict, experiment_options: ExperimentOptions,
    ):
        """From options"""
        del experiment_options
        return cls(**SnakeGameOptions(**config))

    # -- food spawning --------------------------------------------------

    def _roll_lifetime(self) -> float:
        """Return a random lifetime for a new food item [s].

        Uniformly sampled between ``food_lifetime`` and
        ``food_lifetime_max``, then shortened by score.  Returns 0
        when lifetimes are disabled (``food_lifetime <= 0``).
        """
        if self.food_lifetime <= 0:
            return 0.0
        lo = self.food_lifetime
        hi = max(lo, self.food_lifetime_max)
        base = self._rng.uniform(lo, hi) if hi > lo else lo
        return max(1.0, base - 0.5*self.score)

    def _food_scale(self, food: _FoodItem, sim_time: float) -> float:
        """Return the visual scale of a food item based on its age.

        Food starts at full size (1.0) and shrinks linearly to 0 as it
        approaches its expiry time.  Food with no lifetime (``<= 0``)
        always returns 1.0.
        """
        lifetime = food.lifetime
        if lifetime <= 0:
            return 1.0
        age = sim_time - food.spawn_time
        if age >= lifetime:
            return 0.0
        return max(0.0, 1.0 - age / lifetime)

    def _current_min_spawn(self) -> float:
        """Return current min spawn radius, growing with score."""
        return self.min_spawn_radius + 0.02*self.score

    def _spawn_pos(self, com: np.ndarray) -> np.ndarray:
        """Pick a random position near ``com`` respecting exclusion.

        ``com`` is in SI units (meters).  The position is placed on an
        annulus between ``_current_min_spawn()`` and ``spawn_radius``
        at the configured ``spawn_height``.
        """
        min_r = self._current_min_spawn()
        max_r = max(self.spawn_radius, min_r + 0.01)
        angle = self._rng.uniform(0, 2*np.pi)
        radius = self._rng.uniform(min_r, max_r)
        return com + np.array([
            radius*np.cos(angle),
            radius*np.sin(angle),
            self.spawn_height - com[2],
        ])

    def _make_food(self, com: np.ndarray, sim_time: float, is_bad: bool):
        """Create and register a new food item."""
        rgba = self.bad_food_rgba if is_bad else self.food_rgba
        food = _FoodItem(
            pos=self._spawn_pos(com),
            is_bad=is_bad,
            spawn_time=sim_time,
            rgba=list(rgba),
            lifetime=self._roll_lifetime(),
        )
        self.foods.append(food)
        self._draw_food_viewer(food)

    def _respawn_food(self, idx: int, com: np.ndarray, sim_time: float):
        """Replace food at ``idx`` with a fresh spawn of same type."""
        food = self.foods[idx]
        food.pos = self._spawn_pos(com)
        food.spawn_time = sim_time
        food.lifetime = self._roll_lifetime()
        self._update_food_viewer(food, sim_time)

    def _init_foods(self, com: np.ndarray, sim_time: float):
        """(Re)spawn all food items.

        On the first call (from ``initialize_episode``) the viewer
        scene is fresh, so geoms are created.  On subsequent calls
        (game-over reset) existing geoms are reused — only their
        positions are updated — to avoid orphaned geoms accumulating
        in ``viewer.user_scn``.
        """
        n_total = self.n_food + self.n_bad_food
        if len(self.foods) == n_total:
            # Reuse existing food items and their viewer geoms
            for i, food in enumerate(self.foods):
                food.is_bad = i >= self.n_food
                food.rgba = list(
                    self.bad_food_rgba if food.is_bad else self.food_rgba
                )
                food.pos = self._spawn_pos(com)
                food.spawn_time = sim_time
                food.lifetime = self._roll_lifetime()
                self._update_food_viewer(food, sim_time)
        else:
            self.foods.clear()
            for _ in range(self.n_food):
                self._make_food(com, sim_time, is_bad=False)
            for _ in range(self.n_bad_food):
                self._make_food(com, sim_time, is_bad=True)

    # -- rendering ------------------------------------------------------

    def render_food(self, scene, food: _FoodItem, scale: float = 1.0):
        """Add a single food sphere geom to ``scene``.

        ``scale`` shrinks the sphere radius (1.0 = full size).
        """
        radius = self.radius * scale * self.units.meters
        return create_sphere(
            scene,
            size=[radius, 0, 0],
            pos=[p*self.units.meters for p in food.pos],
            rgba=food.rgba,
        )

    def render_scene(self, scene):
        """Render all food spheres onto a scene.

        Called after ``mjv_updateScene`` clears the scene, before
        ``mjr_render``.
        """
        sim_time = getattr(self, '_last_sim_time', 0.0)
        for food in self.foods:
            self.render_food(scene, food, self._food_scale(food, sim_time))

    def _viewer_scn(self):
        """Return the viewer ``user_scn`` if available, else ``None``."""
        viewer = self.viewer
        if viewer is None:
            return None
        return getattr(viewer, 'user_scn', None)

    def _draw_food_viewer(self, food: _FoodItem):
        """Create a persistent geom for ``food`` on the viewer scene.

        Stores the returned geom reference on ``food.geom`` so it can
        be updated in-place when the food moves.
        """
        scn = self._viewer_scn()
        if scn is None:
            return
        food.geom = self.render_food(scn, food)

    def _update_food_viewer(self, food: _FoodItem, sim_time: float = 0.0):
        """Update an existing food geom's position, size, and colour.

        The sphere shrinks linearly toward zero as the food approaches
        its expiry time.  Colour is refreshed so type changes (good ↔
        bad) are visible after a game reset.
        """
        if food.geom is None:
            return
        scale = self._food_scale(food, sim_time)
        radius = self.radius * scale * self.units.meters
        food.geom.pos = [p * self.units.meters for p in food.pos]
        food.geom.size[:3] = radius
        food.geom.rgba = food.rgba

    def _update_display(self, task, sim_time: float = 0.0, force: bool = False):
        """Push the score + energy overlay to the viewer (if any).

        ``set_texts`` acquires the MuJoCo render lock and is expensive
        when called every physics sub-step.  Throttle it to at most
        once per ``_display_dt`` seconds of simulation time, unless
        ``force`` is set (e.g. on game reset or initialisation).
        """
        if not force and sim_time - self._last_display_time < self._display_dt:
            return
        self._last_display_time = sim_time
        viewer = getattr(task, 'viewer', None) or self.viewer
        if viewer is None:
            return
        set_texts = getattr(viewer, 'set_texts', None)
        if set_texts is None:
            return
        if self.max_score > 0:
            score_str = f"Score {self.score}/{self.max_score}"
        else:
            score_str = f"Score {self.score}"
        set_texts([(
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "Snake",
            f"{score_str}  Energy {self.energy:.0f}",
        )])

    # -- game lifecycle -------------------------------------------------

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        """Initialise episode"""
        del physics
        self.units = task.units
        self.viewer = task.viewer
        self.bind_links(task)
        self.score = 0
        self.energy = self.energy_init
        self.foods.clear()  # discard stale geoms (user_scn was reset)
        com = self.com_position(0)
        self._init_foods(com, 0.0)
        self._last_sim_time = 0.0
        self._last_display_time = -1.0
        self._update_display(task, sim_time=0.0, force=True)

    def after_step(self, task: ExperimentTask, physics: Physics):
        """After step"""
        if self.links is None:
            return
        sim_time = physics.time() / task.units.seconds
        dt = task.units.seconds * physics.timestep()
        com = self.com_position(task.iteration - 1)

        # Continuous energy drain
        self.energy -= self.energy_drain * dt
        if self.energy <= 0:
            # Game over -> reset
            self.score = 0
            self.energy = self.energy_init
            self._init_foods(com, sim_time)
            self._update_display(task, sim_time=sim_time, force=True)
            return

        # Check each food for consumption or expiry
        for idx in range(len(self.foods)):
            food = self.foods[idx]
            dist = np.linalg.norm(food.pos[:2] - com[:2])
            if dist < self.catch_distance:
                if food.is_bad:
                    self.score = max(0, self.score - 1)
                    self.energy -= self.energy_bad
                else:
                    self.score += 1
                    self.energy = min(
                        self.energy_max,
                        self.energy + self.energy_recharge,
                    )
                    # Grow the play area slightly
                    self.spawn_radius *= 1.05
                self._respawn_food(idx, com, sim_time)
            else:
                lifetime = food.lifetime
                if lifetime > 0 and sim_time - food.spawn_time > lifetime:
                    self._respawn_food(idx, com, sim_time)

        # Update food geom sizes (shrink toward expiry)
        self._last_sim_time = sim_time
        for food in self.foods:
            self._update_food_viewer(food, sim_time)

        # Clamp energy
        self.energy = max(0.0, min(self.energy_max, self.energy))
        self._update_display(task, sim_time=sim_time)
