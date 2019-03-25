"""Logging"""

import numpy as np
import matplotlib.pyplot as plt


class ExperimentLogger:
    """ExperimentLogger"""

    def __init__(self, model, sim_size):
        super(ExperimentLogger, self).__init__()
        self.sim_size = sim_size
        self.model = model
        self.sensors = SensorsLogger(model, sim_size)
        # [SensorsLogger(model) for sensor in model.sensors]
        self.motors = MotorsLogger(model, sim_size)
        self.phases = PhasesLogger(model, sim_size)

    def update(self, iteration):
        """Update sensors at iteration"""
        self.sensors.update(iteration)
        self.motors.update(iteration)
        self.phases.update(iteration)

    def plot_all(self, sim_times):
        """Plot all"""
        self.sensors.plot_contacts(sim_times)
        self.sensors.plot_ft(sim_times)
        self.motors.plot_body(sim_times)
        self.motors.plot_legs(sim_times)
        self.phases.plot(sim_times)


class SensorsLogger:
    """Sensors logger"""

    def __init__(self, model, size):
        super(SensorsLogger, self).__init__()
        self.model = model
        self.size = size
        self.contact_forces = np.zeros([
            size,
            *np.shape(model.sensors.contact_forces)
        ])
        self.feet_ft = np.zeros([
            size,
            *np.shape(model.sensors.feet_ft)
        ])
        self.feet = model.sensors.feet

    def update(self, iteration):
        """Update sensors logs"""
        self.contact_forces[iteration, :] = self.model.sensors.contact_forces
        self.feet_ft[iteration, :, :] = self.model.sensors.feet_ft

    def plot_contacts(self, times):
        """Plot sensors"""
        # Plot contacts
        plt.figure("Contacts")
        for foot_i, foot in enumerate(self.feet):
            plt.plot(
                times,
                self.contact_forces[:len(times), foot_i],
                label=foot
            )
            plt.xlabel("Time [s]")
            plt.ylabel("Reaction force [N]")
            plt.grid(True)
            plt.legend()

    def plot_ft(self, times):
        """Plot force-torque sensors"""
        # Plot Feet forces
        plt.figure("Feet forces")
        for dim in range(3):
            plt.plot(
                times,
                self.feet_ft[:len(times), 0, dim],
                label=["x", "y", "z"][dim]
            )
            plt.xlabel("Time [s]")
            plt.ylabel("Force [N]")
            plt.grid(True)
            plt.legend()


class MotorsLogger:
    """Motors logger"""

    def __init__(self, model, size):
        super(MotorsLogger, self).__init__()
        self.model = model
        self.size = size
        self.joints_cmds_body = np.zeros([
            size,
            *np.shape(model.motors.joints_cmds_body)
        ])
        self.joints_commanded_body = model.motors.joints_commanded_body
        self.joints_cmds_legs = np.zeros([
            size,
            *np.shape(model.motors.joints_cmds_legs)
        ])
        self.joints_commanded_legs = model.motors.joints_commanded_legs

    def update(self, iteration):
        """Update motor logs"""
        self.joints_cmds_body[iteration, :] = (
            self.model.motors.joints_cmds_body
        )
        self.joints_cmds_legs[iteration, :] = (
            self.model.motors.joints_cmds_legs
        )

    def plot_body(self, times):
        """Plot body motors"""
        plt.figure("Body motor torques")
        for joint_i, joint in enumerate(self.joints_commanded_body):
            plt.plot(
                times,
                self.joints_cmds_body[:len(times), joint_i],
                label=joint
            )
            plt.xlabel("Time [s]")
            plt.ylabel("Torque [Nm]")
            plt.grid(True)
            plt.legend()

    def plot_legs(self, times):
        """Plot legs motors"""
        plt.figure("Legs motor torques")
        for joint_i, joint in enumerate(self.joints_commanded_legs):
            plt.plot(
                times,
                self.joints_cmds_legs[:len(times), joint_i],
                label=joint
            )
            plt.xlabel("Time [s]")
            plt.ylabel("Torque [Nm]")
            plt.grid(True)
            plt.legend()


class PhasesLogger:
    """Phases logger"""

    def __init__(self, model, size):
        super(PhasesLogger, self).__init__()
        self.model = model
        self.size = size
        self.phases_log = np.zeros([
            size,
            *np.shape(model.controller.network.phases)
        ])
        self.oscillator_names = [
            "body_{}".format(i)
            for i in range(11)
        ] +  [
            "leg_{}_{}_{}".format(leg_i, side, joint_i)
            for leg_i in range(2)
            for side in ["L", "R"]
            for joint_i in range(3)
        ]

    def update(self, iteration):
        """Update phase logs"""
        self.phases_log[iteration, :] = (
            self.model.controller.network.phases[:, 0]
        )

    def plot(self, times):
        """Plot body phases"""

        for phase_i, phase in enumerate(self.oscillator_names):
            if "body" in phase:
                plt.figure("Oscillator body phases")
            else:
                plt.figure("Oscillator legs phases")
            plt.plot(
                times,
                self.phases_log[:len(times), phase_i],
                label=phase
            )
            plt.xlabel("Time [s]")
            plt.ylabel("Phase [rad]")
            plt.grid(True)
            plt.legend()