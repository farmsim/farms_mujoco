"""Animat data"""

import numpy as np
import matplotlib.pyplot as plt
from .animat_data_cy import (
    AnimatDataCy,
    NetworkParametersCy,
    OscillatorNetworkStateCy,
    OscillatorArrayCy,
    ConnectivityArrayCy,
    JointsArrayCy,
    SensorsCy,
    ContactsArrayCy,
    ProprioceptionArrayCy,
    GpsArrayCy,
    HydrodynamicsArrayCy
)


class AnimatData(AnimatDataCy):
    """Network parameter"""

    def log(self, times, folder, extension):
        """Log"""
        self.state.log(times, folder, "network_state", extension)
        self.network.log(times, folder, extension)
        self.joints.log(times, folder, "joints", extension)
        self.sensors.log(times, folder, extension)

    def plot(self, times):
        """Plot"""
        self.state.plot(times)
        self.sensors.plot(times)


class NetworkParameters(NetworkParametersCy):
    """Network parameter"""

    def log(self, times, folder, extension):
        """Log"""
        for data, name in [
                [self.oscillators, "oscillators"],
                [self.connectivity, "connectivity"],
                [self.contacts_connectivity, "contacts_connectivity"],
                [self.hydro_connectivity, "hydro_connectivity"]
        ]:
            data.log(times, folder, name, extension)


class OscillatorNetworkState(OscillatorNetworkStateCy):
    """Network state"""

    def plot(self, times):
        """Plot"""
        self.plot_phases(times)
        self.plot_amplitudes(times)

    def plot_phases(self, times):
        """Plot phases"""
        plt.figure("Network state phases")
        for data in np.transpose(self.phases_all()):
            plt.plot(times, data[:len(times)])
        plt.xlabel("Times [s]")
        plt.ylabel("Phases [rad]")
        plt.grid(True)

    def plot_amplitudes(self, times):
        """Plot amplitudes"""
        plt.figure("Network state amplitudes")
        for data in np.transpose(self.amplitudes_all()):
            plt.plot(times, data[:len(times)])
        plt.xlabel("Times [s]")
        plt.ylabel("Amplitudes [rad]")
        plt.grid(True)


class OscillatorArray(OscillatorArrayCy):
    """Oscillator array"""


class ConnectivityArray(ConnectivityArrayCy):
    """Connectivity array"""


class JointsArray(JointsArrayCy):
    """Oscillator array"""


class Sensors(SensorsCy):
    """Sensors"""

    def log(self, times, folder, extension):
        """Log"""
        for data, name in [
                [self.contacts, "contacts"],
                [self.proprioception, "proprioception"],
                [self.gps, "gps"],
                [self.hydrodynamics, "hydrodynamics"]
        ]:
            data.log(times, folder, name, extension)

    def plot(self, times):
        """Plot"""
        self.contacts.plot(times)
        self.proprioception.plot(times)
        self.gps.plot(times)
        self.hydrodynamics.plot(times)


class ContactsArray(ContactsArrayCy):
    """Sensor array"""

    def plot(self, times):
        """Plot"""
        self.plot_ground_reaction_forces(times)
        self.plot_friction_forces(times)
        for ori in range(3):
            self.plot_friction_forces_ori(times, ori=ori)
        self.plot_total_forces(times)

    def plot_ground_reaction_forces(self, times):
        """Plot ground reaction forces"""
        plt.figure("Ground reaction forces")
        for sensor_i in range(self.size[1]):
            data = np.asarray(self.reaction_all(sensor_i))
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1)[:len(times)],
                label="Leg_{}".format(sensor_i)
            )
        plt.legend()
        plt.xlabel("Times [s]")
        plt.ylabel("Forces [N]")
        plt.grid(True)

    def plot_friction_forces(self, times):
        """Plot friction forces"""
        plt.figure("Friction forces")
        for sensor_i in range(self.size[1]):
            data = np.asarray(self.friction_all(sensor_i))
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1)[:len(times)],
                label="Leg_{}".format(sensor_i)
            )
        plt.legend()
        plt.xlabel("Times [s]")
        plt.ylabel("Forces [N]")
        plt.grid(True)

    def plot_friction_forces_ori(self, times, ori):
        """Plot friction forces"""
        plt.figure("Friction forces (ori={})".format(ori))
        for sensor_i in range(self.size[1]):
            data = np.asarray(self.friction_all(sensor_i))
            plt.plot(
                times,
                data[:len(times), ori],
                label="Leg_{}".format(sensor_i)
            )
        plt.legend()
        plt.xlabel("Times [s]")
        plt.ylabel("Forces [N]")
        plt.grid(True)

    def plot_total_forces(self, times):
        """Plot contact forces"""
        plt.figure("Contact total forces")
        for sensor_i in range(self.size[1]):
            data = np.asarray(self.total_all(sensor_i))
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1)[:len(times)],
                label="Leg_{}".format(sensor_i)
            )
        plt.legend()
        plt.xlabel("Times [s]")
        plt.ylabel("Forces [N]")
        plt.grid(True)


class ProprioceptionArray(ProprioceptionArrayCy):
    """Proprioception array"""

    def plot(self, times):
        """Plot"""
        self.plot_positions(times)
        self.plot_velocities(times)
        self.plot_forces(times)
        self.plot_torques(times)
        self.plot_motor_torques(times)

    def plot_positions(self, times):
        """Plot ground reaction forces"""
        plt.figure("Joints positions")
        for joint_i in range(self.size[1]):
            plt.plot(
                times,
                np.asarray(self.positions_all())[:len(times), joint_i],
                label="Joint_{}".format(joint_i)
            )
        plt.legend()
        plt.xlabel("Times [s]")
        plt.ylabel("Joint position [rad]")
        plt.grid(True)

    def plot_velocities(self, times):
        """Plot ground reaction forces"""
        plt.figure("Joints velocities")
        for joint_i in range(self.size[1]):
            plt.plot(
                times,
                np.asarray(self.velocities_all())[:len(times), joint_i],
                label="Joint_{}".format(joint_i)
            )
        plt.legend()
        plt.xlabel("Times [s]")
        plt.ylabel("Joint velocity [rad/s]")
        plt.grid(True)

    def plot_forces(self, times):
        """Plot ground reaction forces"""
        plt.figure("Joints forces")
        for joint_i in range(self.size[1]):
            data = np.linalg.norm(np.asarray(self.forces_all()), axis=-1)
            plt.plot(
                times,
                data[:len(times), joint_i],
                label="Joint_{}".format(joint_i)
            )
        plt.legend()
        plt.xlabel("Times [s]")
        plt.ylabel("Joint force [N]")
        plt.grid(True)

    def plot_torques(self, times):
        """Plot ground reaction torques"""
        plt.figure("Joints torques")
        for joint_i in range(self.size[1]):
            data = np.linalg.norm(np.asarray(self.torques_all()), axis=-1)
            plt.plot(
                times,
                data[:len(times), joint_i],
                label="Joint_{}".format(joint_i)
            )
        plt.legend()
        plt.xlabel("Times [s]")
        plt.ylabel("Joint torque [Nm]")
        plt.grid(True)

    def plot_motor_torques(self, times):
        """Plot ground reaction forces"""
        plt.figure("Joints motor torques")
        for joint_i in range(self.size[1]):
            plt.plot(
                times,
                np.asarray(self.motor_torques())[:len(times), joint_i],
                label="Joint_{}".format(joint_i)
            )
        plt.legend()
        plt.xlabel("Times [s]")
        plt.ylabel("Joint torque [Nm]")
        plt.grid(True)


class GpsArray(GpsArrayCy):
    """Gps array"""

    def plot(self, times):
        """Plot"""
        self.plot_base_position(times, xaxis=0, yaxis=1)
        self.plot_base_velocity(times)

    def plot_base_position(self, times, xaxis=0, yaxis=1):
        """Plot"""
        plt.figure("GPS position")
        for link_i in range(self.size[1]):
            data = np.asarray(self.urdf_positions())[:len(times), link_i]
            plt.plot(
                data[:, xaxis],
                data[:, yaxis],
                label="Link_{}".format(link_i)
            )
        plt.legend()
        plt.xlabel("Position [m]")
        plt.ylabel("Position [m]")
        plt.axis("equal")
        plt.grid(True)

    def plot_base_velocity(self, times):
        """Plot"""
        plt.figure("GPS velocities")
        for link_i in range(self.size[1]):
            data = np.asarray(self.com_lin_velocities())[:len(times), link_i]
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1),
                label="Link_{}".format(link_i)
            )
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity [m/s]")
        plt.grid(True)


class HydrodynamicsArray(HydrodynamicsArrayCy):
    """Hydrodynamics array"""

    def plot(self, times):
        """Plot"""
        self.plot_forces(times)
        self.plot_torques(times)

    def plot_forces(self, times):
        """Plot"""
        plt.figure("Hydrodynamic forces")
        for link_i in range(self.size[1]):
            data = np.asarray(self.forces())[:len(times), link_i]
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1),
                label="Link_{}".format(link_i)
            )
        plt.xlabel("Time [s]")
        plt.ylabel("Forces [N]")
        plt.grid(True)

    def plot_torques(self, times):
        """Plot"""
        plt.figure("Hydrodynamic torques")
        for link_i in range(self.size[1]):
            data = np.asarray(self.torques())[:len(times), link_i]
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1),
                label="Link_{}".format(link_i)
            )
        plt.xlabel("Time [s]")
        plt.ylabel("Torques [Nm]")
        plt.grid(True)
