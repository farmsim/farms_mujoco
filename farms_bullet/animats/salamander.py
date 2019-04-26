"""Salamander"""

import time
import numpy as np

import pybullet

from .animat import Animat
from .model import SalamanderModel
from ..plugins.swimming import viscous_swimming
from ..sensors.sensor import (
    Sensors,
    JointsStatesSensor,
    ContactSensor,
    LinkStateSensor
)


class Salamander(Animat):
    """Salamander animat"""

    def __init__(self, options, timestep, n_iterations):
        super(Salamander, self).__init__(options)
        self.model = None
        self.timestep = timestep
        self.sensors = None
        self.n_iterations = n_iterations

    def spawn(self):
        """Spawn"""
        self.model = SalamanderModel.spawn(
            self.n_iterations,
            self.timestep,
            **self.options
        )
        self._identity = self.model.identity

    def add_sensors(self, arena_identity):
        """Add sensors"""
        # Sensors
        self.sensors = Sensors()
        # Contacts
        self.sensors.add({
            "contact_{}".format(i): ContactSensor(
                self.n_iterations,
                self._identity, self.links[foot],
                arena_identity, -1
            )
            for i, foot in enumerate(self.model.feet)
        })
        # Joints
        n_joints = pybullet.getNumJoints(self._identity)
        self.sensors.add({
            "joints": JointsStatesSensor(
                self.n_iterations,
                self._identity,
                np.arange(n_joints),
                enable_ft=True
            )
        })
        # Base link
        self.sensors.add({
            "base_link": LinkStateSensor(
                self.n_iterations,
                self._identity,
                0,  # Base link
            )
        })

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

    def animat_sensors(self, sim_step):
        """Animat sensors update"""
        tic_sensors = time.time()
        # self.model.sensors.update(
        #     sim_step,
        #     identity=self.identity,
        #     links=[self.links[foot] for foot in self.model.feet],
        #     joints=[
        #         self.joints[joint]
        #         for joint in self.model.sensors.joints_sensors
        #     ]
        # )
        self.sensors.update(sim_step)
        # # Commands
        # self.model.motors.update(
        #     identity=self.identity,
        #     joints_body=[
        #         self.joints[joint]
        #         for joint in self.model.motors.joints_commanded_body
        #     ],
        #     joints_legs=[
        #         self.joints[joint]
        #         for joint in self.model.motors.joints_commanded_legs
        #     ]
        # )
        return time.time() - tic_sensors

    def animat_control(self):
        """Control animat"""
        # Control
        tic_control = time.time()
        self.model.controller.control()
        time_control = time.time() - tic_control
        return time_control

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

    # def animat_logging(self, sim_step):
    #     """Animat logging"""
    #     # Contacts during walking
    #     tic_log = time.time()
    #     self.logger.update(sim_step-1)
    #     return time.time() - tic_log
