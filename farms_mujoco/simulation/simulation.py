"""Simulation"""

import os
import time
from typing import Callable, Union

import pybullet
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import farms_pylog as pylog
from ..model.animat import Animat
from ..model.control import control_models
from ..model.model import SimulationModels
from ..interface.interface import Interfaces
from ..utils.output import redirect_output
from .simulator import init_engine, real_time_handing
from .options import SimulationOptions
from .render import rendering


def simulation_profiler(func: Callable) -> Callable:
    """Profile simulation"""
    def inner(self, profile: bool = False, show_progress: bool = False):
        """Inner function"""
        if profile:
            logger = pybullet.startStateLogging(
                loggingType=pybullet.STATE_LOGGING_PROFILE_TIMINGS,
                fileName='profile.log'
            )
        pbar = tqdm(total=self.options.n_iterations) if show_progress else None
        result = func(self, pbar=pbar)
        if profile:
            pybullet.stopStateLogging(loggingId=logger)
        return result
    return inner


class Simulation:
    """Simulation

    Handles the start/run/end of the experiment, the GUI if not headless, the
    physics properties, etc.
    """

    def __init__(
            self,
            models: SimulationModels,
            options: SimulationOptions,
            interface: Union[Interfaces, None] = None,
    ):
        super().__init__()

        self.models = models
        self.options = options
        self.analytics: tuple = tuple()

        # Initialise engine
        pylog.debug('Initialising physics engine')
        init_engine(self.options.headless, self.options.opengl2)
        if not self.options.headless:
            pylog.debug('Disabling rendering')
            rendering(0)

        # Initialise physics
        pylog.debug('Initialising physics')
        self.init_physics()

        # Initialise models
        pylog.debug('Spawning models')
        self.models.spawn()

        # Simulation
        self.iteration = 0
        self.simulation_state = None

        # Interface
        self.interface = (
            interface
            if interface is not None and not self.options.headless
            else Interfaces()
            if not self.options.headless
            else None
        )
        if not self.options.headless and self.interface is not None:
            self.interface.init_debug(self.options)

        if not self.options.headless:
            pylog.debug('Reactivating rendering')
            rendering(1)

    def save(self):
        """Save experiment state"""
        self.simulation_state = pybullet.saveState()

    def init_physics(self):
        """Initialise physics"""
        # pylog.debug('Resetting simulation')
        # pybullet.resetSimulation()
        pylog.debug('Setting gravity')
        pybullet.setGravity(
            self.options.gravity[0]*self.options.units.gravity,
            self.options.gravity[1]*self.options.units.gravity,
            self.options.gravity[2]*self.options.units.gravity
        )
        pylog.debug('Setting timestep')
        pybullet.setTimeStep(self.options.timestep*self.options.units.seconds)
        pylog.debug('Setting non real-time simulation')
        pybullet.setRealTimeSimulation(0)
        pylog.debug('Setting simulation parameters')
        with redirect_output(pylog.debug):
            pybullet.setPhysicsEngineParameter(
                fixedTimeStep=self.options.timestep*self.options.units.seconds,
                numSolverIterations=self.options.n_solver_iters,
                erp=self.options.erp,
                contactERP=self.options.contact_erp,
                frictionERP=self.options.friction_erp,
                numSubSteps=self.options.num_sub_steps,
                maxNumCmdPer1ms=self.options.max_num_cmd_per_1ms,
                solverResidualThreshold=self.options.residual_threshold,
                constraintSolverType={
                    'si': pybullet.CONSTRAINT_SOLVER_LCP_SI,
                    'dantzig': pybullet.CONSTRAINT_SOLVER_LCP_DANTZIG,
                    'pgs': pybullet.CONSTRAINT_SOLVER_LCP_PGS,
                }[self.options.lcp],
                globalCFM=self.options.cfm,
                reportSolverAnalytics=self.options.report_solver_analytics,
                # jointFeedbackMode=pybullet.JOINT_FEEDBACK_IN_JOINT_FRAME,
                # warmStartingFactor=0,
            )
        pylog.debug('Physics parameters:\n%s', '\n'.join([
            '- {}: {}'.format(key, value)
            for key, value in pybullet.getPhysicsEngineParameters().items()
        ]))

    def check_quit(self):
        """Check quit"""
        if not self.options.headless:
            keys = pybullet.getKeyboardEvents()
            if ord('q') in keys:
                return True
        return False

    @simulation_profiler
    def run(self, pbar=None):
        """Run simulation"""
        while self.iteration < self.options.n_iterations:
            if self.check_quit():
                break
            if self.pre_step(self.iteration):
                self.step(self.iteration)
                self.control(self.iteration)
                self.analytics = pybullet.stepSimulation()
                self.post_step(self.iteration)
                self.iteration += 1
                if pbar is not None:
                    pbar.update(1)

    @simulation_profiler
    def iterator(self, pbar=None):
        """Run simulation"""
        while self.iteration < self.options.n_iterations:
            if self.check_quit():
                break
            if self.pre_step(self.iteration):
                self.step(self.iteration)
                self.control(self.iteration)
                yield self.iteration
                self.analytics = pybullet.stepSimulation()
                # if self.analytics:
                #     print(
                #         self.analytics[0]['numIterationsUsed'],
                #         self.analytics[0]['remainingResidual'],
                #     )
                self.post_step(self.iteration)
                self.iteration += 1
                if pbar is not None:
                    pbar.update(1)

    def pre_step(self, iteration: int) -> bool:
        """Pre-step"""
        assert iteration >= 0
        return True

    def step(self, iteration: int):
        """Step function"""

    def control(self, iteration: int):
        """Physics step"""
        control_models(
            iteration=iteration,
            time=self.options.timestep*iteration,
            timestep=self.options.timestep,
            models=self.models,
            units=self.options.units,
        )

    def post_step(self, iteration: int):
        """Post-step"""

    @staticmethod
    def end():
        """Terminate simulation"""
        with redirect_output(pylog.debug):
            pybullet.disconnect()


class AnimatSimulation(Simulation):
    """Animat simulation"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        animat: Animat = self.animat()

        # Interface
        if not self.options.headless:
            self.interface.init_camera(
                target_identity=(
                    animat.identity()
                    if not self.options.free_camera
                    else None
                ),
                timestep=self.options.timestep,
                rotating_camera=self.options.rotating_camera,
                pitch=self.options.video_pitch,
                yaw=self.options.video_yaw,
                distance=self.options.video_distance,
                motion_filter=self.options.video_filter,
            )

        if self.options.record:
            skips = int(2e-2/self.options.timestep)  # 50 fps
            self.interface.init_video(
                target_identity=animat.identity(),
                simulation_options=self.options,
                pitch=self.options.video_pitch,
                yaw=self.options.video_yaw,
                motion_filter=10*skips*self.options.timestep,
                rotating_camera=self.options.rotating_camera,
            )

        # Real-time handling
        self.tic_rt: npt.ArrayLike = np.zeros(3)

        # Simulation state - Can be saved using self.save()
        self.simulation_state = None

    def animat(self) -> Animat:
        """Salamander animat"""
        return self.models[0]

    def pre_step(self, iteration: int) -> bool:
        """Pre-step

        Returns bool to indicate if simulation should be played or paused
        """
        play = True
        # if not(iteration % 10000) and iteration > 0:
        #     pybullet.restoreState(self.simulation_state)
        #     state = self.animat().data.state
        #     state.array[self.animat().data.iteration] = (
        #         state.default_initial_state()
        #     )
        if not self.options.headless and self.interface is not None:
            play = self.interface.user_params.play().value
            if not iteration % int(0.1/self.options.timestep):
                self.interface.user_params.update()
                self.interface.camera.update()
            if self.interface.user_params.zoom().changed or not iteration:
                self.interface.camera.set_zoom(
                    self.interface.user_params.zoom().value
                )
            if not play:
                time.sleep(0.1)
                self.interface.user_params.update()
            elif self.options.record:
                self.interface.video.record(iteration)
        return play

    def post_step(self, iteration: int):
        """Post step"""
        if (
                not self.options.headless
                and not self.options.fast
                and self.interface is not None
                and self.interface.user_params.rtl().value < 2.99
        ):
            real_time_handing(
                timestep=self.options.timestep,
                tic_rt=self.tic_rt,
                rtl=self.interface.user_params.rtl().value,
            )

    def postprocess(
            self,
            iteration: int,
            log_path: str = '',
            plot: bool = False,
            video: str = '',
            **kwargs,
    ):
        """Plot after simulation"""
        animat = self.animat()
        times = np.arange(
            0,
            self.options.timestep*self.options.n_iterations,
            self.options.timestep
        )[:iteration]

        # Log
        if log_path:
            pylog.info('Saving data to %s', log_path)
            animat.data.to_file(
                os.path.join(log_path, 'simulation.hdf5'),
                iteration,
            )
            self.options.save(os.path.join(log_path, 'simulation_options.yaml'))
            animat.options.save(os.path.join(log_path, 'animat_options.yaml'))

        # Plot
        if plot:
            animat.data.plot(times)

        # Record video
        if video and self.interface is not None:
            self.interface.video.save(
                video,
                iteration=iteration,
                writer=kwargs.pop('writer', 'ffmpeg')
            )
