"""Simulation"""

import os
import time
import pybullet
from tqdm import tqdm
import numpy as np
import farms_pylog as pylog
from ..model.control import control_models
from ..interface.interface import Interfaces
from ..utils.output import redirect_output
from .simulator import init_engine, real_time_handing
from .render import rendering


def simulation_profiler(func):
    """Profile simulation"""
    def inner(self, profile=False, show_progress=False):
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

    def __init__(self, models, options, interface=None):
        super(Simulation, self).__init__()

        self.models = models
        self.options = options
        self.analytics = tuple()

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
        if not self.options.headless:
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
                globalCFM=1e-10,
                reportSolverAnalytics=1,
                # solverResidualThreshold=1e-12,
                # restitutionVelocityThreshold=1e-3,
                # useSplitImpulse=False,
                # splitImpulsePenetrationThreshold=1e-5,
                # contactBreakingThreshold=1e-5
                # numSubSteps=100,
                # maxNumCmdPer1ms=int(1e5),

                # # Parameters
                # fixedTimeStep
                # numSolverIterations
                # useSplitImpulse
                # splitImpulsePenetrationThreshold
                # numSubSteps
                # collisionFilterMode
                # contactBreakingThreshold
                # maxNumCmdPer1ms
                # enableFileCaching
                # restitutionVelocityThreshold
                # erp
                # contactERP
                # frictionERP
                # enableConeFriction
                # deterministicOverlappingPairs
                # solverResidualThreshold
            )
        pylog.debug('Physics parameters:\n{}'.format(
            '\n'.join([
                '- {}: {}'.format(key, value)
                for key, value in pybullet.getPhysicsEngineParameters().items()
            ])
        ))

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
                self.iteration += 1
                self.post_step(self.iteration)
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
                self.analytics = pybullet.stepSimulation()
                # if self.analytics:
                #     print(
                #         self.analytics[0]['numIterationsUsed'],
                #         self.analytics[0]['remainingResidual'],
                #     )
                self.iteration += 1
                self.post_step(self.iteration)
                yield self.iteration-1
                if pbar is not None:
                    pbar.update(1)

    @staticmethod
    def pre_step(_iteration):
        """Pre-step

        Returns bool

        """
        return True

    def step(self, iteration):
        """Step function"""

    def control(self, iteration):
        """Physics step"""
        control_models(
            iteration=iteration,
            time=self.options.timestep*iteration,
            timestep=self.options.timestep,
            models=self.models,
            torques=self.options.units.torques,
        )

    def post_step(self, iteration):
        """Post-step"""

    @staticmethod
    def end():
        """Terminate simulation"""
        # Disconnect from simulation
        with redirect_output(pylog.debug):
            pybullet.disconnect()


class AnimatSimulation(Simulation):
    """Animat simulation"""

    def __init__(self, **kwargs):
        super(AnimatSimulation, self).__init__(**kwargs)
        animat = self.animat()

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
                # fps=1./(skips*self.options.timestep),
                pitch=self.options.video_pitch,
                yaw=self.options.video_yaw,
                # skips=skips,
                motion_filter=10*skips*self.options.timestep,
                # distance=1,
                rotating_camera=self.options.rotating_camera,
                # top_camera=self.options.top_camera
            )

        # Real-time handling
        self.tic_rt = np.zeros(3)

        # Simulation state
        self.simulation_state = None
        self.save()

    def animat(self):
        """Salamander animat"""
        return self.models[0]

    def pre_step(self, iteration):
        """New step"""
        play = True
        # if not(iteration % 10000) and iteration > 0:
        #     pybullet.restoreState(self.simulation_state)
        #     state = self.animat().data.state
        #     state.array[self.animat().data.iteration] = (
        #         state.default_initial_state()
        #     )
        if not self.options.headless:
            play = self.interface.user_params.play().value
            if not iteration % int(0.1/self.options.timestep):
                self.interface.user_params.update()
            if not play:
                time.sleep(0.1)
                self.interface.user_params.update()
        return play

    def post_step(self, iteration):
        """Post step"""

        # Camera
        if not self.options.headless:
            self.interface.camera.update()
            # Camera zoom
            if self.interface.user_params.zoom().changed:
                self.interface.camera.set_zoom(
                    self.interface.user_params.zoom().value
                )
        if self.options.record:
            self.interface.video.record(iteration)

        # Real-time
        if not self.options.headless:
            if (
                    not self.options.fast
                    and self.interface.user_params.rtl().value < 2.99
            ):
                real_time_handing(
                    self.options.timestep,
                    self.tic_rt,
                    rtl=self.interface.user_params.rtl().value
                )

    def postprocess(
            self,
            iteration,
            log_path='',
            plot=False,
            video='',
            **kwargs
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
            pylog.info('Saving data to {}'.format(log_path))
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
        if video:
            self.interface.video.save(
                video,
                iteration=iteration,
                writer=kwargs.pop('writer', 'ffmpeg')
            )
