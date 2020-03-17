"""Sensor logging"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pybullet

from .sensors import (
    JointsStatesSensor,
    ContactsSensors,
    ContactSensor,
    LinksStatesSensor,
    LinkStateSensor
)
from ..animats.amphibious.sensors import AmphibiousGPS


def global2local(vector_global, orientation):
    """Vector in global frame to local frame"""
    orientation_inv = np.array(
        pybullet.getMatrixFromQuaternion(orientation)
    ).reshape([3, 3]).T
    return np.dot(orientation_inv, vector_global)


class SensorLogger:
    """Sensor logger"""

    def __init__(self, sensor):
        super(SensorLogger, self).__init__()
        self._model = sensor

    def array(self):
        """Log array"""
        return self._model.array

    def plot(self, times, figure=None, label=None):
        """Plot"""
        if figure is not None:
            plt.figure(figure)
        for array in self.array().T:
            plt.plot(times, array[:len(times)], label=label)
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

    def plot_array(self, times, array_id, label=None):
        """Plot array"""
        n_sensors = np.shape(self.array())[1]
        for sensor in range(n_sensors):
            plt.plot(
                times,
                self.array()[:len(times), sensor, array_id],
                label=(
                    label + "_" if label is not None else ""
                    + "sensor_{}".format(sensor)
                )
            )
        plt.grid(True)
        plt.legend()

    def plot_array_norm(self, times, array_ids, label=None):
        """Plot array"""
        n_sensors = np.shape(self.array())[1]
        for sensor in range(n_sensors):
            array = np.array(self.array())[:len(times), sensor, array_ids]
            array_norm = np.sqrt(np.sum(array**2, axis=1))
            plt.plot(
                times,
                array_norm,
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
        self.plot_array(times, 0, label=label)
        plt.xlabel("Time [s]")
        plt.ylabel("Position [rad]")

    def plot_velocities(self, times, figure=None, label=None):
        """Plot velocities"""
        if figure is not None:
            plt.figure(figure+"_velocities")
        self.plot_array(times, 1, label=label)
        plt.xlabel("Time [s]")
        plt.ylabel("Angular velocity [rad/s]")

    def plot_forces(self, times, figure=None, label=None):
        """Plot forces"""
        if figure is not None:
            plt.figure(figure+"_forces")
        self.plot_array_norm(times, [2, 3, 4], label=label)
        plt.xlabel("Times [s]")
        plt.ylabel("Force norm [N]")

    def plot_torques(self, times, figure=None, label=None):
        """Plot torques"""
        if figure is not None:
            plt.figure(figure+"_torques")
        self.plot_array_norm(times, [5, 6, 7], label=label)
        plt.xlabel("Times [s]")
        plt.ylabel("Torque norm [N]")

    def plot_torque_cmds(self, times, figure=None, label=None):
        """Plot torque commands"""
        if figure is not None:
            plt.figure(figure+"_torque_cmds")
        self.plot_array(times, 8, label=label)
        plt.xlabel("Time [s]")
        plt.ylabel("Torque [Nm]")


class ContactsLogger(SensorLogger):
    """Joints states logger"""

    def plot(self, times, figure=None, label=None):
        """Plot"""
        for sensor in range(np.shape(self.array())[1]):
            self.plot_normal_force(sensor, times, figure, label=label)
            self.plot_lateral_force(sensor, times, figure, label=label)

    def plot_normal_force(self, sensor, times, figure=None, label=None):
        """Plot normal force"""
        if figure is None:
            figure = "Contact"
        plt.figure(figure+"_{}_normal".format(sensor))
        label = "" if label is None else (label + "_")
        labels = [label + lab for lab in ["x", "y", "z"]]
        for i, array in enumerate(self.array()[:, sensor, :3].T):
            plt.plot(times, array[:len(times)], label=labels[i])
        plt.xlabel("Time [s]")
        plt.ylabel("Normal force [N]")
        plt.legend()
        plt.grid(True)

    def plot_lateral_force(self, sensor, times, figure=None, label=None):
        """Plot lateral force"""
        if figure is None:
            figure = "Contact"
        plt.figure(figure+"_{}_lateral".format(sensor))
        label = "" if label is None else (label + "_")
        labels = [label + lab for lab in ["x", "y", "z"]]
        for i, array in enumerate(self.array()[:, sensor, 3:6].T):
            plt.plot(times, array[:len(times)], label=labels[i])
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.legend()
        plt.grid(True)


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
        for i, array in enumerate(self.array()[:, :3].T):
            plt.plot(times, array[:len(times)], label=labels[i])
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
        for i, array in enumerate(self.array()[:, 3:].T):
            plt.plot(times, array[:len(times)], label=labels[i])
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.legend()
        plt.grid(True)


class LinksStatesLogger(SensorLogger):
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
        self.plot_trajectories_top(
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

    def plot_array(self, times, array_ids, figure=None, labels=None):
        """Plot array"""
        if figure is not None:
            plt.figure(figure)
        for array_i, array_id in enumerate(array_ids):
            plt.plot(
                times,
                self.array()[:len(times), 0, array_id],
                label=labels[array_i]
            )
        plt.grid(True)
        plt.legend()

    def plot_local_array(self, times, array, figure=None, **kwargs):
        """Plot linear velocity in local frame"""
        if figure is not None:
            plt.figure(figure)
        array_local = np.array([
            global2local(array[i], self.array()[i, 0, 10:14])
            for i, _ in enumerate(times)
        ]).T
        labels = kwargs.pop("labels", ["x", "y", "z"])
        for array_i, array_id in enumerate(array_local):
            plt.plot(
                times,
                array_id,
                label=labels[array_i]
            )
        plt.grid(True)
        plt.legend()

    def plot_positions(self, times, **kwargs):
        """Plot positions"""
        figure = kwargs.pop("figure", "") + "_position"
        label = kwargs.pop("label", "pos")
        self.plot_array(
            times=times,
            array_ids=[0, 1, 2],
            figure=figure,
            labels=[label + "_" + model for model in ["x", "y", "z"]]
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Position [m]")

    def plot_trajectory_top(self, times, **kwargs):
        """Plot positions"""
        plt.figure(kwargs.pop("figure", "") + "_trajectory_top")
        plt.plot(
            self.array()[:len(times), 0, 0],
            self.array()[:len(times), 0, 1]
        )
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("Position x [m]")
        plt.ylabel("Position y [m]")

    def plot_trajectories_top(self, times, **kwargs):
        """Plot positions"""
        plt.figure(kwargs.pop("figure", "") + "_trajectories_top")
        shape = np.shape(self.array())
        plt.plot(
            self.array()[:len(times), 0, 0],
            self.array()[:len(times), 0, 1],
            "bo"
        )
        plt.plot(
            self.array()[:len(times), 0, 7],
            self.array()[:len(times), 0, 8],
            "ro"
        )
        for i in range(shape[1]):
            plt.plot(
                self.array()[:len(times), i, 0],
                self.array()[:len(times), i, 1],
                label="link_com_{}".format(i)
            )
            plt.plot(
                self.array()[:len(times), i, 7],
                self.array()[:len(times), i, 8],
                label="link_urdf_{}".format(i)
            )
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("Position x [m]")
        plt.ylabel("Position y [m]")
        plt.legend()

    def plot_linear_velocities(self, times, local=False, **kwargs):
        """Plot velocities"""
        figure = kwargs.pop("figure", "") + "_linear_velocity"
        label = kwargs.pop("label", "pos")
        if local:
            self.plot_local_array(
                times=times,
                array=self.array()[:, 0, 14:17],
                figure=figure,
                labels=[label + "_" + model for model in ["x", "y", "z"]]
            )
        else:
            self.plot_array(
                times=times,
                array_ids=[14, 15, 16],
                figure=figure,
                labels=[label + "_" + model for model in ["x", "y", "z"]]
            )
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity [m/s]")

    def plot_angular_velocities(self, times, local=False, **kwargs):
        """Plot velocities"""
        figure = kwargs.pop("figure", "") + "_angular_velocity"
        label = kwargs.pop("label", "pos")
        if local:
            self.plot_local_array(
                times=times,
                array=self.array()[:, 0, 17:20],
                figure=figure,
                labels=[label + "_" + model for model in ["x", "y", "z"]]
            )
        else:
            self.plot_array(
                times=times,
                array_ids=[17, 18, 19],
                figure=figure,
                labels=[label + "_" + model for model in ["x", "y", "z"]]
            )
        plt.xlabel("Time [s]")
        plt.ylabel("Angular velocity [rad/s]")


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

    def plot_array(self, times, array_ids, figure=None, labels=None):
        """Plot array"""
        if figure is not None:
            plt.figure(figure)
        for array_i, array_id in enumerate(array_ids):
            plt.plot(
                times,
                self.array()[:len(times), array_id],
                label=labels[array_i]
            )
        plt.grid(True)
        plt.legend()

    def plot_local_array(self, times, array, figure=None, **kwargs):
        """Plot linear velocity in local frame"""
        if figure is not None:
            plt.figure(figure)
        array_local = np.array([
            global2local(array[i], self.array()[i, 10:14])
            for i, _ in enumerate(times)
        ]).T
        labels = kwargs.pop("labels", ["x", "y", "z"])
        for array_i, array_id in enumerate(array_local):
            plt.plot(
                times,
                array_id,
                label=labels[array_i]
            )
        plt.grid(True)
        plt.legend()

    def plot_positions(self, times, **kwargs):
        """Plot positions"""
        figure = kwargs.pop("figure", "") + "_position"
        label = kwargs.pop("label", "pos")
        self.plot_array(
            times=times,
            array_ids=[0, 1, 2],
            figure=figure,
            labels=[label + "_" + model for model in ["x", "y", "z"]]
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Position [m]")

    def plot_trajectory_top(self, times, **kwargs):
        """Plot positions"""
        plt.figure(kwargs.pop("figure", "") + "_trajectory_top")
        plt.plot(
            self.array()[:len(times), 0],
            self.array()[:len(times), 1]
        )
        plt.grid(True)
        plt.xlabel("Position x [m]")
        plt.ylabel("Position y [m]")

    def plot_linear_velocities(self, times, local=False, **kwargs):
        """Plot velocities"""
        figure = kwargs.pop("figure", "") + "_linear_velocity"
        label = kwargs.pop("label", "pos")
        if local:
            self.plot_local_array(
                times=times,
                array=self.array()[:, 7:10],
                figure=figure,
                labels=[label + "_" + model for model in ["x", "y", "z"]]
            )
        else:
            self.plot_array(
                times=times,
                array_ids=[7, 8, 9],
                figure=figure,
                labels=[label + "_" + model for model in ["x", "y", "z"]]
            )
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity [m/s]")

    def plot_angular_velocities(self, times, local=False, **kwargs):
        """Plot velocities"""
        figure = kwargs.pop("figure", "") + "_angular_velocity"
        label = kwargs.pop("label", "pos")
        if local:
            self.plot_local_array(
                times=times,
                array=self.array()[:, 10:],
                figure=figure,
                labels=[label + "_" + model for model in ["x", "y", "z"]]
            )
        else:
            self.plot_array(
                times=times,
                array_ids=[10, 11, 12],
                figure=figure,
                labels=[label + "_" + model for model in ["x", "y", "z"]]
            )
        plt.xlabel("Time [s]")
        plt.ylabel("Angular velocity [rad/s]")


class SensorsLogger(dict):
    """Sensors logging"""

    mapping = {
        JointsStatesSensor: JointsStatesLogger,
        ContactsSensors: ContactsLogger,
        ContactSensor: ContactLogger,
        AmphibiousGPS: LinksStatesLogger,
        LinksStatesSensor: LinksStatesLogger,
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

    def log_all(self, times, folder, extension="npy"):
        """Log all sensors logs"""
        os.makedirs(folder, exist_ok=True)
        if extension == "npy":
            save_function = np.save
            nosplit = True
        elif extension in ("txt", "csv"):
            save_function = np.savetxt
            nosplit = False
        else:
            raise Exception(
                "Format {} is not valid for logging array".format(extension)
            )
        save_function(folder+"/times."+extension, times)
        for sensor_name, sensor in self._sensors.items():
            if nosplit or self[sensor].array.ndim == 2:
                path = folder + "/" + sensor_name + "." + extension
                save_function(path, self[sensor].array[:len(times)])
            elif self[sensor].array.ndim == 3:
                for i in range(np.shape(self[sensor].array)[1]):
                    path = folder+"/"+sensor_name+"_{}.".format(i)+extension
                    save_function(path, self[sensor].array[:len(times), i])
            else:
                msg = "Dimensionality {} is not valid for extension of type {}"
                raise Exception(msg.format(self[sensor].array.ndim, extension))
