"""Simon's experiment"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pybullet

from .experiment import Experiment
from ..animats.model_options import ModelOptions
from ..animats.simon import SimonAnimat
from ..arenas.arena import FlooredArena
from ..sensors.sensor import (
    Sensors,
    JointsStatesSensor,
    ContactSensor,
    LinkStateSensor
)
from ..sensors.logging import SensorsLogger


class SimonExperiment(Experiment):
    """Simon experiment"""

    def __init__(self, sim_options, n_iterations, **kwargs):
        self.animat_options = kwargs.pop("animat_options", ModelOptions())
        super(SimonExperiment, self).__init__(
            animat=SimonAnimat(
                self.animat_options,
                sim_options.timestep,
                n_iterations
            ),
            arena=FlooredArena(),
            timestep=sim_options.timestep,
            n_iterations=n_iterations
        )
        self.logger = None

    def spawn(self):
        """Spawn"""
        self._spawn()

        # # Feet constraints - Closed chain
        # print("ATTEMPTING TO INSERT CONSTRAINT")
        # feet_positions = np.array([
        #     [0.1, 0.08, 0.01],
        #     [0.1, -0.08, 0.01],
        #     [-0.1, 0.08, 0.01],
        #     [-0.1, -0.08, 0.01]
        # ])
        # cid = [None for _ in feet_positions]
        # for i, pos in enumerate(feet_positions):
        #     cid[i] = pybullet.createConstraint(
        #         self.arena.floor.identity, -1,
        #         self.animat.identity, 1 + 2*i,
        #         pybullet.JOINT_POINT2POINT,  # JOINT_PRISMATIC,  # JOINT_POINT2POINT
        #         [0.0, 0.0, 1.0],
        #         [0.0, 0.0, 0.0],
        #         [0.0, 0.0, 0.0]
        #     )
        #     pybullet.changeConstraint(cid[i], maxForce=1e5)
        # print("CONSTRAINT INSERTED")

        # Sensors
        n_joints = pybullet.getNumJoints(self.animat.identity)
        self.animat.sensors = Sensors()
        # Contacts
        self.animat.sensors.add({
            "contact_{}".format(i): ContactSensor(
                self.n_iterations,
                self.animat.identity, 1+2*i,
                self.arena.floor.identity, -1
            )
            for i in range(4)
        })
        # Joints
        self.animat.sensors.add({
            "joints": JointsStatesSensor(
                self.n_iterations,
                self.animat.identity,
                np.arange(n_joints),
                enable_ft=True
            )
        })
        # Base link
        self.animat.sensors.add({
            "base_link": LinkStateSensor(
                self.n_iterations,
                self.animat.identity,
                0,  # Base link
            )
        })

        # logger
        self.logger = SensorsLogger(self.animat.sensors)

    def pre_step(self, sim_step):
        """New step"""
        return True

    def step(self, sim_step):
        """Step"""
        # for sensor in self.animat.sensors:
        #     sensor.update()
        self.animat.sensors.update(sim_step)
        # contacts_sensors = [
        #     self.animat.sensors["contact_{}".format(i)].get_normal_force()
        #     for i in range(4)
        # ]
        # print("Sensors contact forces: {}".format(contacts_sensors))
        # self.logger[sim_step, :] = contacts_sensors
        self.logger.update_logs(sim_step)
        n_joints = pybullet.getNumJoints(self.animat.identity)
        # pybullet.setJointMotorControlArray(
        #     self.animat.identity,
        #     np.arange(n_joints),
        #     pybullet.TORQUE_CONTROL,
        #     forces=0.2*np.ones(n_joints)
        # )
        target_positions = np.zeros(n_joints)
        target_velocities = np.zeros(n_joints)
        joint_control = int((1e-3 * sim_step) % n_joints)
        _sim_step = sim_step % 1000
        target_positions[joint_control] = (
            0.3*(_sim_step-100)/300 if 100 < _sim_step < 400
            else -0.3*(_sim_step-600)/300 if 600 < _sim_step < 900
            else 0
        )
        joints_states = pybullet.getJointStates(
            self.animat.identity,
            np.arange(n_joints)
        )
        joints_positions = np.array([
            joints_states[joint][0]
            for joint in range(n_joints)
        ])
        joints_velocity = np.array([
            joints_states[joint][1]
            for joint in range(n_joints)
        ])
        pybullet.setJointMotorControlArray(
            self.animat.identity,
            np.arange(n_joints),
            pybullet.TORQUE_CONTROL,
            forces=(
                1e1*(target_positions - joints_positions)
                - 1e-2*joints_velocity
            )
        )
        # pybullet.setJointMotorControlArray(
        #     self.animat.identity,
        #     np.arange(n_joints),
        #     pybullet.POSITION_CONTROL,
        #     targetPositions=target_positions,
        #     targetVelocities=target_velocities,
        #     forces=10*np.ones(n_joints)
        # )
        pybullet.stepSimulation()
        sim_step += 1
        time.sleep(1e-3)
