"""Salamander"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pybullet

from .experiment import Experiment
from ..animats.salamander import Salamander
from ..animats.model_options import ModelOptions
from ..arenas.arena import FlooredArena
from ..interface.interface import Interfaces
from ..profile.profile import SimulationProfiler
from ..simulations.simulator import real_time_handing
# from ..loggers.logging import ExperimentLogger
from ..sensors.logging import SensorsLogger


class SalamanderExperiment(Experiment):
    """Salamander experiment"""

    def __init__(self, sim_options, n_iterations, **kwargs):
        self.animat_options = kwargs.pop("animat_options", ModelOptions())
        super(SalamanderExperiment, self).__init__(
            animat=Salamander(
                self.animat_options,
                sim_options.timestep,
                n_iterations
            ),
            arena=FlooredArena(),
            timestep=sim_options.timestep,
            n_iterations=n_iterations
        )
        self.sim_options = sim_options
        self.interface = Interfaces()
        self.simulation_state = None
        self.profile = SimulationProfiler(self.sim_options.duration)
        self.forces_torques = np.zeros([n_iterations, 2, 10, 3])
        self.logger = None

    def spawn(self):
        """Spawn"""
        # Elements
        self._spawn()
        self.animat.add_sensors(self.arena.floor.identity)
        self.logger = SensorsLogger(self.animat.sensors)
        # Interface
        if not self.sim_options.headless:
            self.interface.init_camera(
                target_identity=self.animat.identity,
                timestep=self.timestep,
                rotating_camera=self.sim_options.rotating_camera,
                top_camera=self.sim_options.top_camera
            )
            self.interface.init_debug(animat_options=self.animat_options)
        if self.sim_options.record and not self.sim_options.headless:
            self.interface.init_video(
                target_identity=self.animat.identity,
                timestep=self.timestep*25,
                size=self.n_iterations//25,
                rotating_camera=self.sim_options.rotating_camera,
                top_camera=self.sim_options.top_camera
            )

    def save(self):
        """Save experiment state"""
        self.simulation_state = pybullet.saveState()

    def pre_step(self, sim_step, play=True):
        """New step"""
        if not(sim_step % 10000) and sim_step > 0:
            pybullet.restoreState(self.simulation_state)
            network = self.animat.model.controller.network
            network.state.array[network.iteration] = (
                network.state.default_initial_state()
            )
        if not self.sim_options.headless:
            play = self.interface.user_params.play.value
            if not sim_step % 100:
                self.interface.user_params.update()
            if not play:
                time.sleep(0.5)
                self.interface.user_params.update()
        return play

    def step(self, sim_step):
        """Simulation step"""
        self.tic_rt = time.time()
        self.sim_time = self.timestep*sim_step
        # Animat sensors
        time_sensors = self.animat.animat_sensors(sim_step)
        self.profile.sensors_time += time_sensors
        if sim_step < self.n_iterations-1:
            if not self.sim_options.headless:
                self.animat_interface()
            # Plugins
            external_forces = self.animat.animat_physics()
            if external_forces is not None:
                self.forces_torques[sim_step] = external_forces
            self.time_plugin = time.time() - self.tic_rt
            self.profile.plugin_time += self.time_plugin
            # Control animat
            time_control = self.animat.animat_control()
            self.profile.ctrl_time += time_control
            # Interface
            # Physics
            self.tic_sim = time.time()
            pybullet.stepSimulation()
            sim_step += 1
            self.toc_sim = time.time()
            self.profile.physics_time += self.toc_sim - self.tic_sim
            # Camera
            tic_camera = time.time()
            if not self.sim_options.headless:
                if self.sim_options.record and not sim_step % 25:
                    self.camera_record.record(sim_step//25-1)
                # User camera
                if (
                        not sim_step % self.interface.camera_skips
                        and not self.sim_options.free_camera
                ):
                    self.interface.camera.update()
            self.profile.camera_time += time.time() - tic_camera
            # Real-time
            self.toc_rt = time.time()
            tic_rt = time.time()
            if (
                    not self.sim_options.fast
                    and self.interface.user_params.rtl.value < 3
            ):
                real_time_handing(
                    self.timestep, self.tic_rt, self.toc_rt,
                    rtl=self.interface.user_params.rtl.value,
                    time_plugin=self.time_plugin,
                    time_sim=self.toc_sim-self.tic_sim,
                    time_control=time_control
                )
            self.profile.waitrt_time = time.time() - tic_rt

    def animat_interface(self):
        """Animat interface"""
        # Control
        if self.interface.user_params.gait.changed:
            self.animat_options.gait = self.interface.user_params.gait.value
            self.animat.model.controller.update_gait(
                self.animat_options.gait,
                self.animat.joints,
                self.timestep
            )
            pybullet.setGravity(
                0, 0, -1e-2 if self.animat_options.gait == "swimming" else -9.81
            )
            self.interface.user_params.gait.changed = False
        if self.interface.user_params.frequency.changed:
            network = self.animat.model.controller.network
            network.parameters.oscillators.freqs = (
                self.interface.user_params.frequency.value
            )
            self.interface.user_params.frequency.changed = False
        if self.interface.user_params.body_offset.changed:
            network = self.animat.model.controller.network
            network.parameters.joints.set_body_offset(
                self.interface.user_params.body_offset.value
            )
            self.interface.user_params.body_offset.changed = False
        if (
                self.interface.user_params.drive_speed.changed
                or self.interface.user_params.drive_turn.changed
        ):
            self.animat.model.controller.network.update_drive(
                self.interface.user_params.drive_speed.value,
                self.interface.user_params.drive_turn.value
            )
            self.interface.user_params.drive_speed.changed = False
            self.interface.user_params.drive_turn.changed = False

    def postprocess(self, plot=None, log_path=None, log_extension=None):
        """Plot after simulation"""
        # Plot
        if plot:
            self.logger.plot_all(self.times_simulated)
            plt.show()
        if log_path:
            self.logger.log_all(
                self.times_simulated,
                folder=log_path,
                extension=log_extension
            )

        # Record video
        if self.sim_options.record and not self.sim_options.headless:
            self.camera_record.save("video.avi")

    def end(self, sim_step, sim_time):
        """Terminate experiment"""
        self.profile.sim_duration = self.timestep*sim_step
        self.profile.sim_time = sim_time
        self.profile.print_times()
