"""Salamander"""

import time
from .animat import Animat
from .model import SalamanderModel
from ..loggers.logging import ExperimentLogger
from ..plugins.swimming import viscous_swimming


class Salamander(Animat):
    """Salamander animat"""

    def __init__(self, options, timestep, n_iterations):
        super(Salamander, self).__init__(options)
        self.model = None
        self.timestep = timestep
        self.logger = None
        self.n_iterations = n_iterations

    def spawn(self):
        """Spawn"""
        self.model = SalamanderModel.spawn(
            self.n_iterations,
            self.timestep,
            **self.options
        )
        self._identity = self.model.identity
        self.logger = ExperimentLogger(
            self.model,
            self.n_iterations
        )

    @property
    def links(self):
        """Links"""
        return self.model.links

    @property
    def joints(self):
        """Joints"""
        return self.model.joints

    def step(self):
        """Step"""
        self.animat_physics()
        self.animat_control()

    def log(self):
        """Log"""
        self.animat_logging()

    def animat_physics(self):
        """Animat physics"""
        # Swimming
        forces = None
        if self.options.gait == "swimming":
            forces = viscous_swimming(
                self.identity,
                self.links
            )
        return forces

    def animat_control(self):
        """Control animat"""
        # Control
        tic_control = time.time()
        self.model.controller.control()
        time_control = time.time() - tic_control
        return time_control

    def animat_logging(self, sim_step):
        """Animat logging"""
        # Contacts during walking
        tic_sensors = time.time()
        self.model.sensors.update(
            sim_step,
            identity=self.identity,
            links=[self.links[foot] for foot in self.model.feet],
            joints=[
                self.joints[joint]
                for joint in self.model.sensors.joints_sensors
            ]
        )
        # Commands
        self.model.motors.update(
            identity=self.identity,
            joints_body=[
                self.joints[joint]
                for joint in self.model.motors.joints_commanded_body
            ],
            joints_legs=[
                self.joints[joint]
                for joint in self.model.motors.joints_commanded_legs
            ]
        )
        time_sensors = time.time() - tic_sensors
        tic_log = time.time()
        self.logger.update(sim_step-1)
        time_log = time.time() - tic_log
        return time_sensors, time_log
