"""Camera"""

import os
from dataclasses import dataclass

import mujoco
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
try:
    import cv2
    USE_CV2 = True
except ImportError:
    USE_CV2 = False

from farms_core import pylog
from farms_core.doc import ChildDoc, ExtensionDoc
from farms_core.options import Options
from farms_core.experiment.options import ExperimentOptions
from farms_core.simulation.extensions import TaskExtension


@dataclass
class VideoWriterOptions:
    """Class for keeping track of an item in inventory."""
    path: str
    file_extension: str
    writer: str


class CameraRecordingOptions(Options):
    """Camera recording options"""

    @classmethod
    def doc(cls):
        """Doc"""
        return ExtensionDoc(
            name="Video recording options",
            description="Describes the video recording options.",
            class_type=cls,
            children=[
                ChildDoc(
                    name="path",
                    class_type=str,
                    description=(
                        "Path to where the video should be saved. Empty string"
                        " to disable recording."
                    ),
                ),
                ChildDoc(
                    name="fps",
                    class_type=float,
                    description="Video framerate.",
                ),
                ChildDoc(
                    name="speed",
                    class_type=float,
                    description=(
                        "Speed factor at which the video should be played."
                    ),
                ),
                ChildDoc(
                    name="resolution",
                    class_type="list[int]",
                    description="Video resolution (e.g. [1280, 720]).",
                ),
                ChildDoc(
                    name="animat_id",
                    class_type=int,
                    description=(
                        "Either the animat index or null for fixed camera."
                    ),
                ),
                ChildDoc(
                    name="offset",
                    class_type=float,
                    description=(
                        "Video position offset with respect to origin"
                        " or animat (depending on animat_id)."
                    ),
                ),
                ChildDoc(
                    name="azimuth",
                    class_type=float,
                    description="Video yaw angle in degrees.",
                ),
                ChildDoc(
                    name="elevation",
                    class_type=float,
                    description="Video elevation angle in degrees.",
                ),
                ChildDoc(
                    name="distance",
                    class_type=float,
                    description="Camera distance from animat.",
                ),
                ChildDoc(
                    name="angular_velocity",
                    class_type=float,
                    description=(
                        "Angular velocity at which camera rotates around focus"
                        " point in [deg/s]"
                    ),
                ),
                ChildDoc(
                    name="motion_filter",
                    class_type=float,
                    description="Video motion filter.",
                ),
            ],
        )

    def __init__(self, **kwargs):
        super().__init__()
        self.path: str = kwargs.pop('path')
        self.fps: float = kwargs.pop('fps', 30)
        self.speed: float = kwargs.pop('speed', 1.0)
        self.resolution: list[int] = kwargs.pop('resolution', [1280, 720])
        self.camera: None | int = kwargs.pop('camera', None)
        self.animat_id: int = kwargs.pop('animat_id', 0)
        self.offset: float = kwargs.pop('offset', [0, 0, 0])
        self.azimuth: float = kwargs.pop('azimuth', 0)
        self.elevation: float = kwargs.pop('elevation', -15)
        self.distance: float = kwargs.pop('distance', 2)
        self.angular_velocity = kwargs.pop('angular_velocity', 0)
        self.geomgroups: list[int] = kwargs.pop('geomgroups', [0, 1, 0, 1, 0, 0])
        assert not kwargs, kwargs


class CameraRecording(TaskExtension):
    """Camera recording extension"""

    def __init__(
            self,
            timestep: float,
            n_iterations: int,
            fps: float = 30,
            speed: float = 1.0,
            **kwargs,
    ):
        super().__init__()
        self.renderer = None
        self.last_capture = 0
        self.speed = speed
        self.timestep = timestep / speed
        self.n_iterations = n_iterations
        self.camera = kwargs.pop('camera', None)
        self.offset: float = kwargs.pop('offset', [0, 0, 0])
        self.animat_id = kwargs.pop('animat_id', 0)
        self.distance = kwargs.pop('distance', 2)
        self.elevation = kwargs.pop('elevation', -15)
        self.azimuth = kwargs.pop('azimuth', 0)
        self.angular_velocity = kwargs.pop('angular_velocity', 0)  # [deg/s]
        self.geomgroups = kwargs.pop('geomgroups', [1, 1, 0, 1, 0, 0])
        self.motion_filter = kwargs.pop('motion_filter', 10*timestep)
        self.width, self.height = kwargs.pop('resolution', [640, 480])
        self.skips = kwargs.pop('skips', max(0, int(speed/(timestep*fps))-1))
        self.fps = 1/(self.timestep*(self.skips+1))
        self.sample = 0
        self.viewer: str = kwargs.pop('viewer', 'MuJoCo')
        self.links = None
        self.render_options = None
        self.data = np.zeros(
            [n_iterations//(self.skips+1)+1, self.height, self.width, 3],
            dtype=np.uint8
        )
        self.out = None
        video_path, video_extension = os.path.splitext(kwargs.pop('path'))
        match video_extension:
            case 'mp4':
                writer = 'ffmpeg'
            case 'html':
                writer = 'html'
            case _:
                pylog.warning(
                    'Unknown write for "%s", trying with ffmpeg',
                    video_extension,
                )
                pylog.warning(
                    'Options for Matplotlib include %s',
                    manimation.writers.list()
                )
                writer = 'ffmpeg'
        self.video = VideoWriterOptions(
            path=video_path,
            file_extension=video_extension,
            writer=writer,
        )

    @classmethod
    def from_options(cls, config: dict, experiment_options: ExperimentOptions):
        """From options"""
        sim_options = experiment_options.simulation
        return cls(
            timestep=sim_options.physics.timestep,
            n_iterations=sim_options.runtime.n_iterations,
            viewer=sim_options.mujoco.viewer,
            **CameraRecordingOptions(**config),
        )

    def initialize_episode(self, task, physics):
        """Initialize episode"""
        self.sample = 0
        self.data = np.zeros(
            [self.n_iterations//(self.skips+1)+1, self.height, self.width, 3],
            dtype=np.uint8,
        )
        if USE_CV2:
            self.out = cv2.VideoWriter(
                f'{self.video.path}{self.video.file_extension}',
                cv2.VideoWriter_fourcc(*'mp4v'),  # X264
                self.fps,
                (self.width, self.height),
            )
        if self.viewer != 'dm_control':
            if self.animat_id is not None:
                self.links = task.data.animats[self.animat_id].sensors.links
            self.render_options = mujoco.MjvOption()
            mujoco.mjv_defaultOption(self.render_options)
            self.render_options.geomgroup = self.geomgroups
            if self.camera is None:
                self.camera = mujoco.MjvCamera()
                self.camera.type      = mujoco.mjtCamera.mjCAMERA_FREE
                self.camera.lookat[:] = np.array(self.offset)
                if self.links is not None:
                    self.camera.lookat[:] += np.array(
                        self.links.global_com_position(iteration=0),
                    )
                self.camera.distance = self.distance
                self.camera.azimuth = self.azimuth
                self.camera.elevation = self.elevation
                self.renderer = mujoco.Renderer(
                    physics.model.ptr,
                    width=self.width,
                    height=self.height,
                )

    def before_step(self, task, action, physics):
        """Before step"""
        del action
        if not task.iteration % (self.skips+1):
            if self.viewer == 'dm_control':
                self.data[self.sample, :, :, :] = physics.render(
                    width=self.width,
                    height=self.height,
                    camera_id=self.camera,
                )
            else:
                now = physics.time()/task.units.seconds
                timediff = now - self.last_capture
                self.last_capture = now
                self.camera.azimuth += self.angular_velocity*timediff
                self.camera.lookat[:] = np.array(self.offset)
                if self.links is not None:
                    self.camera.lookat[:] += np.array(
                        self.links.global_com_position(
                            iteration=task.iteration-1,
                        ),
                    )
                self.camera.distance  = self.distance
                self.camera.elevation = self.elevation
                if self.renderer is not None:
                    self.renderer.update_scene(
                        physics.data.ptr,
                        camera=self.camera,
                        scene_option=self.render_options,
                    )
                    self.renderer.render(out=self.data[self.sample, :, :, :])
            if self.out is not None:
                self.out.write(self.data[self.sample, :, :, :][:, :, [2, 1, 0]])
            self.sample += 1

    def end_episode(self, task, physics):
        """End episode"""
        del physics
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.save(
            filename=f'{self.video.path}{self.video.file_extension}',
            iteration=task.iteration,
            writer=self.video.writer,
        )

    def save(
            self,
            filename: str = 'video.avi',
            iteration: int | None = None,
            writer: str = 'ffmpeg',
    ):
        """Save recording"""
        if iteration is not None:
            assert iteration//(self.skips+1) <= self.sample, (
                f'{iteration//(self.skips+1)} !<= {self.sample}'
            )
        sample = (
            iteration//(self.skips+1)
            if iteration is not None
            else self.sample
        )
        if USE_CV2:
            data = np.zeros(
                [sample+1, self.height, self.width, 4],
                dtype=np.uint8
            )
            data[:, :, :, :3] = self.data[:sample+1]
        else:
            data = self.data
        pylog.debug(
            'Recording video to %s with %s (fps=%s, speed=%s, skips=%s, frame=%s/%s)',
            filename,
            writer,
            self.fps,
            self.speed,
            self.skips,
            iteration//(self.skips+1) if iteration is not None else self.sample,
            self.sample,
        )
        if self.out is not None:
            self.out.release()
            path = f'{self.video.path}{self.video.file_extension}'
            pylog.info(f"Video saved to {path}")
        else:
            metadata = {
                'title': 'FARMS simulation',
                'artist': 'FARMS',
                'comment': 'FARMS simulation',
            }
            ffmpegwriter = manimation.writers[writer]
            writer = ffmpegwriter(fps=self.fps, metadata=metadata)
            size = 10
            fig = plt.figure(
                'Recording',
                figsize=(size, size*self.height/self.width)
            )
            fig_ax = plt.gca()
            dirname = os.path.dirname(filename)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            pylog.info(f"Saving video to {filename}")
            with writer.saving(fig, filename, dpi=self.width/size):
                ims = None
                for frame in tqdm(data):
                    ims = render_matplotlib_image(fig_ax, frame, ims=ims)
                    writer.grab_frame()
            plt.close(fig)


def render_matplotlib_image(fig_ax, img, ims=None, cbar_label='', clim=None):
    """Render matplotlib image"""
    if ims is None:
        ims = plt.imshow(img)
        fig_ax.spines['top'].set_visible(False)
        fig_ax.spines['right'].set_visible(False)
        fig_ax.spines['bottom'].set_visible(False)
        fig_ax.spines['left'].set_visible(False)
        fig_ax.get_xaxis().set_visible(False)
        fig_ax.get_yaxis().set_visible(False)
        fig_ax.get_xaxis().set_ticks([])
        fig_ax.get_yaxis().set_ticks([])
        fig_ax.set_aspect(aspect=1)
        plt.axis('off')
        plt.tight_layout(pad=0)
        if cbar_label:
            divider = make_axes_locatable(fig_ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(ims, cax=cax)
            cbar.set_label(cbar_label, rotation=90)
        if clim:
            plt.clim(clim)
    else:
        ims.set_data(img)
    return ims
