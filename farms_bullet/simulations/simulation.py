"""Simulation"""

import os
import pickle

import numpy as np
import pybullet
from tqdm import tqdm

from .simulator import init_engine
from ..render.render import rendering


class SimulationElements(dict):
    """Simulation elements"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, animat, arena):
        super(SimulationElements, self).__init__()
        self.animat = animat
        self.arena = arena

    def spawn(self):
        """Spawn"""
        for element_name, element in self.items():
            print("Spawning {}".format(element_name))
            element.spawn()

    def step(self):
        """Step"""
        for element in self.values():
            element.step()

    def log(self):
        """Step"""
        for element in self.values():
            element.log()


class Simulation:
    """Simulation

    Handles the start/run/end of the experiment, the GUI if not headless, the
    physics properties, etc.

    """

    def __init__(self, elements, options):
        super(Simulation, self).__init__()

        self.elements = elements
        self.options = options

        # Initialise engine
        print("Initialising physics engine")
        init_engine(self.options.headless)
        if not self.options.headless:
            print("Disabling rendering")
            rendering(0)

        # Initialise physics
        print("Initialising physics")
        self.init_physics()

        # Initialise models
        print("Spawning elements")
        self.elements.spawn()

        # Simulation
        self.iteration = 0
        self.simulation_state = None

        # Interface
        self.interface = None

        if not self.options.headless:
            print("Reactivating rendering")
            rendering(1)

    def save(self):
        """Save experiment state"""
        self.simulation_state = pybullet.saveState()

    def init_physics(self):
        """Initialise physics"""
        # print("Resetting simulation")
        # pybullet.resetSimulation()
        print("Setting gravity")
        pybullet.setGravity(0, 0, -9.81*self.options.units.gravity)
        print("Setting timestep")
        pybullet.setTimeStep(self.options.timestep*self.options.units.seconds)
        print("Setting non real-time simulation")
        pybullet.setRealTimeSimulation(0)
        print("Setting simulation parameters")
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
        print("Physics parameters:\n{}".format(
            pybullet.getPhysicsEngineParameters()
        ))

    def pre_step(self, sim_step):
        """Pre-step"""
        raise NotImplementedError

    def run(self, profile=False, show_progress=False, yield_data=False):
        """Run simulation"""
        # Run simulation
        if profile:
            logger = pybullet.startStateLogging(
                loggingType=pybullet.STATE_LOGGING_PROFILE_TIMINGS,
                fileName="profile.log"
            )
        if show_progress:
            pbar = tqdm(total=self.options.n_iterations)
        while self.iteration < self.options.n_iterations:
            if not self.options.headless:
                keys = pybullet.getKeyboardEvents()
                if ord("q") in keys:
                    break
            if self.pre_step(self.iteration):
                self.step(self.iteration)
                self.iteration += 1
                if yield_data:
                    yield self.iteration, self.elements.animat.data
            if show_progress:
                pbar.update(1)
        if profile:
            pybullet.stopStateLogging(loggingId=logger)

    def postprocess(self, iteration, **kwargs):
        """Plot after simulation"""
        times = np.arange(
            0,
            self.options.timestep*self.options.n_iterations,
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
            self.elements.animat.data.log(
                times,
                folder=log_path,
                extension=log_extension
            )
            print(self.options)
            with open(log_path+"/simulation_options.pickle", "wb") as options:
                pickle.dump(self.options, options)
            with open(log_path+"/simulation_options.pickle", "rb") as options:
                test = pickle.load(options)
                print("Wrote simulation options:\n{}".format(test))
            with open(log_path+"/animat_options.pickle", "wb") as options:
                pickle.dump(self.elements.animat.options, options)
            with open(log_path+"/animat_options.pickle", "rb") as options:
                test = pickle.load(options)
                print("Wrote animat options:\n{}".format(test))

        # Plot
        plot = kwargs.pop("plot", None)
        if plot:
            self.elements.animat.data.plot(times)

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
