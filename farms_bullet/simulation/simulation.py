"""Simulation"""

import os
import pickle

import numpy as np
import pybullet
from tqdm import tqdm

import farms_pylog as pylog

from ..model.control import control_models
from .simulator import init_engine
from .render import rendering


def simulation_profiler(func):
    """Profile simulation"""
    def inner(self, profile=False, show_progress=False):
        """Inner function"""
        if profile:
            logger = pybullet.startStateLogging(
                loggingType=pybullet.STATE_LOGGING_PROFILE_TIMINGS,
                fileName="profile.log"
            )
        pbar = tqdm(total=self.options.n_iterations()) if show_progress else None
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

    def __init__(self, models, options):
        super(Simulation, self).__init__()

        self.models = models
        self.options = options

        # Initialise engine
        pylog.debug("Initialising physics engine")
        init_engine(self.options.headless)
        if not self.options.headless:
            pylog.debug("Disabling rendering")
            rendering(0)

        # Initialise physics
        pylog.debug("Initialising physics")
        self.init_physics()

        # Initialise models
        pylog.debug("Spawning models")
        self.models.spawn()

        # Simulation
        self.iteration = 0
        self.simulation_state = None

        # Interface
        self.interface = None

        if not self.options.headless:
            pylog.debug("Reactivating rendering")
            rendering(1)

    def save(self):
        """Save experiment state"""
        self.simulation_state = pybullet.saveState()

    def init_physics(self):
        """Initialise physics"""
        # pylog.debug("Resetting simulation")
        # pybullet.resetSimulation()
        pylog.debug("Setting gravity")
        pybullet.setGravity(
            self.options.gravity[0]*self.options.units.gravity,
            self.options.gravity[1]*self.options.units.gravity,
            self.options.gravity[2]*self.options.units.gravity
        )
        pylog.debug("Setting timestep")
        pybullet.setTimeStep(self.options.timestep*self.options.units.seconds)
        pylog.debug("Setting non real-time simulation")
        pybullet.setRealTimeSimulation(0)
        pylog.debug("Setting simulation parameters")
        pybullet.setPhysicsEngineParameter(
            fixedTimeStep=self.options.timestep*self.options.units.seconds,
            numSolverIterations=self.options.n_solver_iters,
            erp=1e-2,
            contactERP=0,
            frictionERP=0,
            numSubSteps=0,
            maxNumCmdPer1ms=int(1e8),
            solverResidualThreshold=0,
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
        pylog.debug("Physics parameters:\n{}".format(
            pybullet.getPhysicsEngineParameters()
        ))

    def check_quit(self):
        """Check quit"""
        if not self.options.headless:
            keys = pybullet.getKeyboardEvents()
            if ord("q") in keys:
                return True
        return False

    @simulation_profiler
    def run(self, pbar=None):
        """Run simulation"""
        while self.iteration < self.options.n_iterations():
            if self.check_quit():
                break
            if self.pre_step(self.iteration):
                self.step(self.iteration)
                self.control_step(self.iteration)
                pybullet.stepSimulation()
                self.iteration += 1
                self.post_step(self.iteration)
                if pbar is not None:
                    pbar.update(1)

    @simulation_profiler
    def iterator(self, pbar=None):
        """Run simulation"""
        while self.iteration < self.options.n_iterations():
            if self.check_quit():
                break
            if self.pre_step(self.iteration):
                self.step(self.iteration)
                self.control_step(self.iteration)
                pybullet.stepSimulation()
                self.iteration += 1
                self.post_step(self.iteration)
                yield self.iteration-1
                if pbar is not None:
                    pbar.update(1)

    def pre_step(self, sim_step):
        """Pre-step

        Returns bool

        """

    def step(self, sim_step):
        """Step function"""

    def control_step(self, sim_step):
        """Physics step"""
        control_models(
            iteration=sim_step,
            models=self.models,
            seconds=self.options.units.seconds,
            torques=self.options.units.torques,
        )

    def post_step(self, sim_step):
        """Post-step"""

    def postprocess(self, iteration, **kwargs):
        """Plot after simulation"""
        times = np.arange(
            0,
            self.options.timestep*self.options.n_iterations(),
            self.options.timestep
        )[:iteration]

        # Log
        log_path = kwargs.pop("log_path", None)
        if log_path:
            log_extension = kwargs.pop("log_extension", None)
            os.makedirs(log_path, exist_ok=True)
            if log_extension == "npy":
                save_function = np.save
            elif log_extension in ("txt", "csv"):
                save_function = np.savetxt
            else:
                raise Exception(
                    "Format {} is not valid for logging array".format(log_extension)
                )
            save_function(log_path+"/times."+log_extension, times)
            self.models.animat.data.log(
                times,
                folder=log_path,
                extension=log_extension
            )
            pylog.debug(self.options)
            with open(log_path+"/simulation_options.pickle", "wb") as options:
                pickle.dump(self.options, options)
            with open(log_path+"/simulation_options.pickle", "rb") as options:
                test = pickle.load(options)
                pylog.debug("Wrote simulation options:\n{}".format(test))
            with open(log_path+"/animat_options.pickle", "wb") as options:
                pickle.dump(self.models.animat.options, options)
            with open(log_path+"/animat_options.pickle", "rb") as options:
                test = pickle.load(options)
                pylog.debug("Wrote animat options:\n{}".format(test))

        # Plot
        plot = kwargs.pop("plot", None)
        if plot:
            self.models.animat.data.plot(times)

        # Record video
        record = kwargs.pop("record", None)
        if record:
            self.interface.video.save(
                "{}.avi".format(self.options.video_name)
            )

    def end(self):
        """Terminate simulation"""
        # Disconnect from simulation
        pybullet.disconnect()
