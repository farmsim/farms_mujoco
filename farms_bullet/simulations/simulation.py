"""Simulation"""

import time
import numpy as np
import pybullet

from .simulator import init_engine
from ..render.render import rendering


class Simulation:
    """Simulation"""

    def __init__(self, experiment, simulation_options, animat_options):
        super(Simulation, self).__init__()

        # Options
        self.sim_options = simulation_options
        self.animat_options = animat_options

        # Initialise engine
        init_engine(self.sim_options.headless)
        rendering(0)

        # Parameters
        self.timestep = self.sim_options.timestep
        self.times = np.arange(0, self.sim_options.duration, self.timestep)

        # Initialise physics
        self.init_physics()

        # Initialise models
        self.experiment = experiment
        self.experiment.spawn()
        self.animat = self.experiment.animat

        # Simulation
        self.sim_step = 0

        rendering(1)

    def init_physics(self):
        """Initialise physics"""
        gait = self.animat_options.gait
        pybullet.resetSimulation()
        pybullet.setGravity(0, 0, -1e-2 if gait == "swimming" else -9.81)
        pybullet.setTimeStep(self.timestep)
        pybullet.setRealTimeSimulation(0)
        pybullet.setPhysicsEngineParameter(
            fixedTimeStep=self.timestep,
            numSolverIterations=50,
            erp=0,
            contactERP=0,
            frictionERP=0
        )
        print("Physics parameters:\n{}".format(
            pybullet.getPhysicsEngineParameters()
        ))

    def run(self):
        """Run simulation"""
        # Run simulation
        self.tic = time.time()
        loop_time = 0
        play = True
        while self.sim_step < len(self.times):
            if not self.sim_options.headless:
                keys = pybullet.getKeyboardEvents()
                if ord("q") in keys:
                    break
                play = self.experiment.pre_step(self.sim_step)
            if play:
                tic_loop = time.time()
                self.experiment.step(self.sim_step)
                self.sim_step += 1
                # self.experiment.log()
                loop_time += time.time() - tic_loop
        print("Loop time: {} [s]".format(loop_time))
        self.toc = time.time()
        self.experiment.times_simulated = self.times[:self.sim_step]

    def end(self):
        """Terminate simulation"""
        # End experiment
        self.experiment.end(self.sim_step, self.toc - self.tic)
        # Disconnect from simulation
        pybullet.disconnect()
