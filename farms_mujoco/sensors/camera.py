"""Camera"""

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from farms_core import pylog
from farms_mujoco.simulation.task import TaskCallback


class CameraCallback(TaskCallback):
    """Camera callback"""

    def __init__(
            self,
            camera_id,
            timestep: float,
            n_iterations: int,
            fps: float,
            **kwargs,
    ):
        super().__init__()
        self.camera_id = camera_id
        self.timestep = timestep
        self.n_iterations = n_iterations
        self.motion_filter = kwargs.pop('motion_filter', 10*timestep)
        self.width = kwargs.pop('width', 640)
        self.height = kwargs.pop('height', 480)
        self.skips = kwargs.pop('skips', max(0, int(1//(timestep*fps))-1))
        self.fps = 1/(self.timestep*(self.skips+1))
        self.sample = 0
        self.data = np.zeros(
            [n_iterations//(self.skips+1)+1, self.height, self.width, 3],
            dtype=np.uint8
        )

    def initialize_episode(self, task, physics):
        """Initialize episode"""
        self.data = np.zeros(
            [self.n_iterations//(self.skips+1)+1, self.height, self.width, 3],
            dtype=np.uint8,
        )

    def before_step(self, task, action, physics):
        """Step hydrodynamics"""
        if not task.iteration % (self.skips+1):
            # sample = (
            #     task.iteration
            #     if not self.skips
            #     else task.iteration//(self.skips+1)
            # )
            self.data[self.sample, :, :, :] = physics.render(
                width=self.width,
                height=self.height,
                camera_id=self.camera_id,
            )
            self.sample += 1

    def save(
            self,
            filename: str = 'video.avi',
            iteration: int = None,
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
            'Recording video to %s with %s (fps=%s, skips=%s, frame=%s/%s)',
            filename,
            writer,
            self.fps,
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


def save_video(camera, video_path, iteration=None):
    """Save video"""
    if 'ffmpeg' in manimation.writers.list():
        camera.save(
            filename=f'{video_path}.mp4',
            iteration=iteration,
            writer='ffmpeg',
        )
    elif 'html' in manimation.writers.list():
        camera.save(
            filename=f'{video_path}.html',
            iteration=iteration,
            writer='html',
        )
    else:
        pylog.error(
            'No known writers, maybe you can use: %s',
            manimation.writers.list(),
        )
