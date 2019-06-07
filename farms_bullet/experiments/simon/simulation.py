"""Simulation of Simon's experiment"""

import time
import numpy as np
import pybullet

from ...simulations.simulation import Simulation, SimulationElements
# from ..experiments.simon import SimonExperiment
from ...simulations.simulation_options import SimulationOptions
from ...arenas.arena import FlooredArena
from ...sensors.sensors import (
    Sensors,
    JointsStatesSensor,
    ContactsSensors,
    LinksStatesSensor
)
from ...sensors.logging import SensorsLogger

from .animat import SimonAnimat
from .animat_options import SimonOptions


class SimonSimulation(Simulation):
    """Simon experiment simulation"""

    def __init__(self, simulation_options, animat_options):
        super(SimonSimulation, self).__init__(
            elements=SimulationElements(
                animat=SimonAnimat(
                    animat_options,
                    simulation_options.timestep,
                    simulation_options.n_iterations
                ),
                arena=FlooredArena()
            ),
            options=simulation_options
        )
        self.spawn()

    def spawn(self):
        """Spawn"""

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
        #         self.elements.arena.floor.identity, -1,
        #         self.elements.animat.identity, 1 + 2*i,
        #         pybullet.JOINT_POINT2POINT,  # JOINT_PRISMATIC,  # JOINT_POINT2POINT
        #         [0.0, 0.0, 1.0],
        #         [0.0, 0.0, 0.0],
        #         [0.0, 0.0, 0.0]
        #     )
        #     pybullet.changeConstraint(cid[i], maxForce=1e5)
        # print("CONSTRAINT INSERTED")

        # Sensors
        n_joints = pybullet.getNumJoints(self.elements.animat.identity)
        self.elements.animat.sensors = Sensors()
        # Contacts
        # self.elements.animat.sensors.add({
        #     "contact_{}".format(i): ContactSensor(
        #         self.options.n_iterations,
        #         self.elements.animat.identity,
        #         1+2*i
        #     )
        #     for i in range(4)
        # })
        self.elements.animat.sensors.add({
            "contacts": ContactsSensors(
                self.elements.animat.data.sensors.contacts.array,
                [self.elements.animat.identity for _ in range(4)],
                [1+2*i for i in range(4)]
            )
        })
        # Joints
        self.elements.animat.sensors.add({
            "joints": JointsStatesSensor(
                self.elements.animat.data.sensors.proprioception.array,
                self.elements.animat.identity,
                np.arange(n_joints),
                enable_ft=True
            )
        })
        # Base link
        self.elements.animat.sensors.add({
            "base_link": LinksStatesSensor(
                self.elements.animat.data.sensors.gps.array,
                self.elements.animat.identity,
                [["base_link", 0, -1]],  # Base link
            )
        })

        # logger
        self.logger = SensorsLogger(self.elements.animat.sensors)

    def pre_step(self, sim_step):
        """New step"""
        return True

    def step(self, sim_step):
        """Step"""
        # for sensor in self.elements.animat.sensors:
        #     sensor.update()
        self.elements.animat.sensors.update(sim_step)
        # contacts_sensors = [
        #     self.elements.animat.sensors["contact_{}".format(i)].get_normal_force()
        #     for i in range(4)
        # ]
        # print("Sensors contact forces: {}".format(contacts_sensors))
        # self.logger[sim_step, :] = contacts_sensors
        self.logger.update_logs(sim_step)
        n_joints = pybullet.getNumJoints(self.elements.animat.identity)
        # pybullet.setJointMotorControlArray(
        #     self.elements.animat.identity,
        #     np.arange(n_joints),
        #     pybullet.TORQUE_CONTROL,
        #     forces=0.2*np.ones(n_joints)
        # )
        target_positions = np.zeros(n_joints)
        target_velocities = np.zeros(n_joints)
        joint_control = int(5e-4*sim_step) % n_joints
        _sim_step = sim_step % 1000
        sign = 1 if sim_step % 2000 < 1000 else -1
        target_positions[joint_control] = (
            sign*0.3*(_sim_step-100)/400 if 100 < _sim_step < 500
            else -sign*0.3*(_sim_step-900)/400 if 500 < _sim_step < 900
            else 0
        )
        joints_states = pybullet.getJointStates(
            self.elements.animat.identity,
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
            self.elements.animat.identity,
            np.arange(n_joints),
            pybullet.TORQUE_CONTROL,
            forces=(
                1e1*(target_positions - joints_positions)
                + 1e-1*(target_velocities - joints_velocity)
            )
        )
        # pybullet.setJointMotorControlArray(
        #     self.elements.animat.identity,
        #     np.arange(n_joints),
        #     pybullet.POSITION_CONTROL,
        #     targetPositions=target_positions,
        #     targetVelocities=target_velocities,
        #     # forces=10*np.ones(n_joints)
        # )
        pybullet.stepSimulation()
        sim_step += 1
        time.sleep(1e-3)


def run_simon(sim_options=None, animat_options=None):
    """Run Simon's experiment"""

    # Parse command line arguments
    if sim_options is None:
        simulation_options = SimulationOptions.with_clargs(duration=100)
    if animat_options is None:
        animat_options = SimonOptions()

    # Setup simulation
    print("Creating simulation")
    sim = SimonSimulation(
        simulation_options=simulation_options,
        animat_options=animat_options
    )

    # Run simulation
    print("Running simulation")
    sim.run()

    # Show results
    print("Analysing simulation")
    sim.postprocess(
        iteration=sim.iteration,
        plot=simulation_options.plot,
        log_path=simulation_options.log_path,
        log_extension=simulation_options.log_extension,
        record=sim.options.record and not sim.options.headless
    )
    sim.end()
