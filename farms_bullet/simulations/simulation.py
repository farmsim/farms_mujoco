"""Simulation"""

import pickle

import numpy as np
import pybullet

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
        init_engine(self.options.headless)
        rendering(0)

        # Initialise physics
        self.init_physics()

        # Initialise models
        self.elements.spawn()

        # Simulation
        self.iteration = 0
        self.simulation_state = None
        self.logger = NotImplemented

        # Interface
        self.interface = None

        rendering(1)

    def save(self):
        """Save experiment state"""
        self.simulation_state = pybullet.saveState()

    def init_physics(self):
        """Initialise physics"""
        pybullet.resetSimulation()
        pybullet.setGravity(0, 0, -9.81*self.options.units.gravity)
        pybullet.setTimeStep(self.options.timestep*self.options.units.seconds)
        pybullet.setRealTimeSimulation(0)
        pybullet.setPhysicsEngineParameter(
            fixedTimeStep=self.options.timestep*self.options.units.seconds,
            numSolverIterations=self.options.n_solver_iters,
            erp=1e-2,
            contactERP=1e-2,
            frictionERP=1e-2,
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

    def run(self):
        """Run simulation"""
        # Run simulation
        while self.iteration < self.options.n_iterations:
            if not self.options.headless:
                keys = pybullet.getKeyboardEvents()
                if ord("q") in keys:
                    break
            if self.pre_step(self.iteration):
                self.step(self.iteration)
                self.iteration += 1

    def postprocess(self, iteration, **kwargs):
        """Plot after simulation"""
        times = np.arange(
            0,
            self.options.timestep*self.options.n_iterations,
            self.options.timestep
        )[:iteration]

        plot = kwargs.pop("plot", None)
        if plot:
            # self.logger.plot_all(times)
            self.elements.animat.data.plot(times)

        log_path = kwargs.pop("log_path", None)
        if log_path:
            log_extension = kwargs.pop("log_extension", None)
            # self.logger.log_all(
            #     times,
            #     folder=log_path,
            #     extension=log_extension
            # )
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
