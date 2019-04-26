"""Sensor logging"""

import numpy as np
import matplotlib.pyplot as plt
import pybullet

from .sensor import JointsStatesSensor, ContactSensor, LinkStateSensor


def global2local(vector_global, orientation):
    """Vector in global frame to local frame"""
    orientation_inv = np.linalg.inv(np.array(
        pybullet.getMatrixFromQuaternion(orientation)
    ).reshape([3, 3]))
    return np.dot(orientation_inv, vector_global)


class SensorLogger:
    """Sensor logger"""

    def __init__(self, sensor):
        super(SensorLogger, self).__init__()
        self._element = sensor

    @property
    def data(self):
        """Log data"""
        return self._element.data

    def plot(self, times, figure=None, label=None):
        """Plot"""
        if figure is not None:
            plt.figure(figure)
        for data in self.data.T:
            plt.plot(times, data[:len(times)], label=label)
        plt.grid(True)
        plt.legend()


class JointsStatesLogger(SensorLogger):
    """Joints states logger"""

    def plot(self, times, figure=None, label=None):
        """Plot"""
        self.plot_positions(times, figure, label=label)
        self.plot_velocities(times, figure, label=label)
        self.plot_forces(times, figure, label=label)
        self.plot_torques(times, figure, label=label)
        self.plot_torque_cmds(times, figure, label=label)

    def plot_data(self, times, data_id, label=None):
        """Plot data"""
        n_sensors = np.shape(self.data)[1]
        for sensor in range(n_sensors):
            plt.plot(
                times,
                self.data[:len(times), sensor, data_id],
                label=(
                    label + "_" if label is not None else ""
                    + "sensor_{}".format(sensor)
                )
            )
        plt.grid(True)
        plt.legend()

    def plot_data_norm(self, times, data_ids, label=None):
        """Plot data"""
        n_sensors = np.shape(self.data)[1]
        for sensor in range(n_sensors):
            data = self.data[:len(times), sensor, data_ids]
            data_norm = np.sqrt(np.sum(data**2, axis=1))
            plt.plot(
                times,
                data_norm,
                label=(
                    label
                    if label is not None
                    else "sensor"
                    + "_{}".format(sensor)
                )
            )
        plt.grid(True)
        plt.legend()

    def plot_positions(self, times, figure=None, label=None):
        """Plot positions"""
        if figure is not None:
            plt.figure(figure+"_positions")
        self.plot_data(times, 0, label=label)
        plt.xlabel("Time [s]")
        plt.ylabel("Position [rad]")

    def plot_velocities(self, times, figure=None, label=None):
        """Plot velocities"""
        if figure is not None:
            plt.figure(figure+"_velocities")
        self.plot_data(times, 1, label=label)
        plt.xlabel("Time [s]")
        plt.ylabel("Angular velocity [rad/s]")

    def plot_forces(self, times, figure=None, label=None):
        """Plot forces"""
        if figure is not None:
            plt.figure(figure+"_forces")
        self.plot_data_norm(times, [2, 3, 4], label=label)
        plt.xlabel("Times [s]")
        plt.ylabel("Force norm [N]")

    def plot_torques(self, times, figure=None, label=None):
        """Plot torques"""
        if figure is not None:
            plt.figure(figure+"_torques")
        self.plot_data_norm(times, [5, 6, 7], label=label)
        plt.xlabel("Times [s]")
        plt.ylabel("Torque norm [N]")

    def plot_torque_cmds(self, times, figure=None, label=None):
        """Plot torque commands"""
        if figure is not None:
            plt.figure(figure+"_torque_cmds")
        self.plot_data(times, 8, label=label)
        plt.xlabel("Time [s]")
        plt.ylabel("Torque [Nm]")


class ContactLogger(SensorLogger):
    """Joints states logger"""

    def plot(self, times, figure=None, label=None):
        """Plot"""
        self.plot_normal_force(times, figure, label=label)
        self.plot_lateral_force(times, figure, label=label)

    def plot_normal_force(self, times, figure=None, label=None):
        """Plot normal force"""
        if figure is None:
            figure = "Contact"
        plt.figure(figure+"_normal")
        label = "" if label is None else (label + "_")
        labels = [label + lab for lab in ["x", "y", "z"]]
        for i, data in enumerate(self.data[:, :3].T):
            plt.plot(times, data[:len(times)], label=labels[i])
        plt.xlabel("Time [s]")
        plt.ylabel("Normal force [N]")
        plt.legend()
        plt.grid(True)

    def plot_lateral_force(self, times, figure=None, label=None):
        """Plot lateral force"""
        if figure is None:
            figure = "Contact"
        plt.figure(figure+"_lateral")
        label = "" if label is None else (label + "_")
        labels = [label + lab for lab in ["x", "y", "z"]]
        for i, data in enumerate(self.data[:, 3:].T):
            plt.plot(times, data[:len(times)], label=labels[i])
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.legend()
        plt.grid(True)


class LinkStateLogger(SensorLogger):
    """Link state logger"""

    def plot(self, times, figure=None, label=None):
        """Plot"""
        self.plot_positions(
            times=times,
            figure=figure,
            label="pos" if label is None else label
        )
        self.plot_trajectory_top(
            times=times,
            figure=figure
        )
        self.plot_linear_velocities(
            times=times,
            local=True,
            figure=figure+"_local",
            label="linear_vel_local" if label is None else label
        )
        self.plot_linear_velocities(
            times=times,
            local=False,
            figure=figure+"_global",
            label="linal_vel_global" if label is None else label
        )
        self.plot_angular_velocities(
            times=times,
            figure=figure,
            local=True,
            label="angular_vel_local" if label is None else label
        )
        self.plot_angular_velocities(
            times=times,
            figure=figure,
            local=False,
            label="angular_vel_global" if label is None else label
        )

    def plot_data(self, times, data_ids, figure=None, labels=None):
        """Plot data"""
        if figure is not None:
            plt.figure(figure)
        for data_i, data_id in enumerate(data_ids):
            plt.plot(
                times,
                self.data[:len(times), data_id],
                label=labels[data_i]
            )
        plt.grid(True)
        plt.legend()

    def plot_local_data(self, times, data, figure=None, **kwargs):
        """Plot linear velocity in local frame"""
        if figure is not None:
            plt.figure(figure)
        data_local = np.array([
            global2local(data[i], self.data[i, 3:7])
            for i, _ in enumerate(times)
        ]).T
        labels = kwargs.pop("labels", ["x", "y", "z"])
        for data_i, data_id in enumerate(data_local):
            plt.plot(
                times,
                data_id,
                label=labels[data_i]
            )
        plt.grid(True)
        plt.legend()

    def plot_positions(self, times, **kwargs):
        """Plot positions"""
        figure = kwargs.pop("figure", "") + "_position"
        label = kwargs.pop("label", "pos")
        self.plot_data(
            times=times,
            data_ids=[0, 1, 2],
            figure=figure,
            labels=[label + "_" + element for element in ["x", "y", "z"]]
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Position [m]")

    def plot_trajectory_top(self, times, **kwargs):
        """Plot positions"""
        plt.figure(kwargs.pop("figure", "") + "_trajectory_top")
        plt.plot(
            self.data[:len(times), 0],
            self.data[:len(times), 1]
        )
        plt.grid(True)
        plt.xlabel("Position x [m]")
        plt.ylabel("Position y [m]")

    def plot_linear_velocities(self, times, local=False, **kwargs):
        """Plot velocities"""
        figure = kwargs.pop("figure", "") + "_linear_velocity"
        label = kwargs.pop("label", "pos")
        if local:
            self.plot_local_data(
                times=times,
                data=self.data[:, 7:10],
                figure=figure,
                labels=[label + "_" + element for element in ["x", "y", "z"]]
            )
        else:
            self.plot_data(
                times=times,
                data_ids=[7, 8, 9],
                figure=figure,
                labels=[label + "_" + element for element in ["x", "y", "z"]]
            )
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity [m/s]")

    def plot_angular_velocities(self, times, local=False, **kwargs):
        """Plot velocities"""
        figure = kwargs.pop("figure", "") + "_angular_velocity"
        label = kwargs.pop("label", "pos")
        if local:
            self.plot_local_data(
                times=times,
                data=self.data[:, 10:],
                figure=figure,
                labels=[label + "_" + element for element in ["x", "y", "z"]]
            )
        else:
            self.plot_data(
                times=times,
                data_ids=[10, 11, 12],
                figure=figure,
                labels=[label + "_" + element for element in ["x", "y", "z"]]
            )
        plt.xlabel("Time [s]")
        plt.ylabel("Angular velocity [rad/s]")


class SensorsLogger(dict):
    """Sensors logging"""

    mapping = {
        JointsStatesSensor: JointsStatesLogger,
        ContactSensor: ContactLogger,
        LinkStateSensor: LinkStateLogger
    }
    default = SensorLogger

    def __init__(self, sensors):
        self._sensors = sensors
        super(SensorsLogger, self).__init__({
            sensor: self.mapping[type(sensor)](sensor)
            for sensor in self._sensors.values()
        })

    def update_logs(self, sim_step):
        """Update sensors logs"""
        self._sensors.update(sim_step)

    def plot_all(self, times):
        """Plot all sensors logs"""
        for sensor_name, sensor in self._sensors.items():
            figure_name = (
                sensor_name
                if isinstance(sensor, ContactSensor)
                else str(sensor)
            )
            label = (
                sensor_name
                if isinstance(sensor, ContactSensor)
                else None
            )
            self[sensor].plot(times, figure=figure_name, label=label)
