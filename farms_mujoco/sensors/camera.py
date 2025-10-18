"""Camera"""

import os
from dataclasses import dataclass

import mujoco
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from farms_core import pylog
from farms_core.experiment.options import ExperimentOptions
from farms_core.simulation.extensions import TaskExtension


@dataclass
class VideoWriterOptions:
    """Class for keeping track of an item in inventory."""
    path: str
    file_extension: str
    writer: str


class CameraRecordingExtension(TaskExtension):
    """Camera recording extension"""

    def __init__(
            self,
            camera_id,
            timestep: float,
            n_iterations: int,
            fps: float = 30,
            speed: float = 1.0,
            **kwargs,
    ):
        super().__init__()
        self.renderer = None
        self.camera_id = camera_id
        self.speed = speed
        self.timestep = timestep / speed
        self.n_iterations = n_iterations
        self.motion_filter = kwargs.pop('motion_filter', 10*timestep)
        self.width = kwargs.pop('width', 640)
        self.height = kwargs.pop('height', 480)
        self.skips = kwargs.pop('skips', max(0, int(speed//(timestep*fps))-1))
        self.fps = 1/(self.timestep*(self.skips+1))
        self.sample = 0
        self.viewer: str = kwargs.pop('viewer', 'MuJoCo')
        self.data = np.zeros(
            [n_iterations//(self.skips+1)+1, self.height, self.width, 3],
            dtype=np.uint8
        )
        video_path, video_extension = os.path.splitext(kwargs.pop('video_path'))
        match video_extension:
            case 'mp4':
                writer = 'ffmpeg'
            case 'html':
                writer = 'html'
            case _:
                pylog.warning(
                    'Unknown write for "%s", trying with ffmpeg'
                    ' (other options include %s)',
                    video_extension,
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
        del config
        sim_options = experiment_options.simulation
        return cls(
            camera_id=0,
            timestep=sim_options.physics.timestep,
            n_iterations=sim_options.runtime.n_iterations,
            viewer=sim_options.mujoco.viewer,
            fps=sim_options.video.fps,
            width=sim_options.video.resolution[0],
            height=sim_options.video.resolution[1],
            video_path=sim_options.video.path,
        )

    def initialize_episode(self, task, physics):
        """Initialize episode"""
        del task
        self.data = np.zeros(
            [self.n_iterations//(self.skips+1)+1, self.height, self.width, 3],
            dtype=np.uint8,
        )
        if self.viewer != 'dm_control':
            self.render_options = mujoco.MjvOption()
            mujoco.mjv_defaultOption(self.render_options)
            self.render_options.geomgroup[:4] = [1, 1, 0, 1]

    def before_step(self, task, action, physics):
        if not task.iteration % (self.skips+1):
            if self.viewer == 'dm_control':
                self.data[self.sample, :, :, :] = physics.render(
                    width=self.width,
                    height=self.height,
                    camera_id=self.camera_id,
                )
            else:
                with mujoco.Renderer(
                        physics.model.ptr,
                        width=self.width,
                        height=self.height,
                ) as renderer:
                    renderer.update_scene(
                        physics.data.ptr,
                        camera=self.camera_id,
                        scene_option=self.render_options,
                    )
                    renderer.render(out=self.data[self.sample, :, :, :])
            self.sample += 1

    def end_episode(self, task, physics):
        del physics
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
        data = (
            self.data[:iteration//(self.skips+1)]
            if iteration is not None
            else self.data[:self.sample]
        )
        ffmpegwriter = manimation.writers[writer]
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
        metadata = dict(
            title='FARMS simulation',
            artist='FARMS',
            comment='FARMS simulation'
        )
        writer = ffmpegwriter(fps=self.fps, metadata=metadata)
        size = 10
        fig = plt.figure(
            'Recording',
            figsize=(size, size*self.height/self.width)
        )
        fig_ax = plt.gca()
        ims = None
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with writer.saving(fig, filename, dpi=self.width/size):
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
