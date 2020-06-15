"""Animat data"""

import numpy as np
import matplotlib.pyplot as plt
from .array import DoubleArray3D
from .data_cy import (
    SensorsDataCy,
    ContactsArrayCy,
    ProprioceptionArrayCy,
    GpsArrayCy,
    HydrodynamicsArrayCy
)


NPDTYPE = np.float64
NPUITYPE = np.uintc


def to_array(array, iteration=None):
    """To array or None"""
    if array is not None:
        array = np.array(array)
        if iteration is not None:
            array = array[:iteration]
    return array


class SensorData(DoubleArray3D):
    """SensorData"""

    def __init__(self, array, names):
        super(SensorData, self).__init__(array)
        self.names = names

    @classmethod
    def from_dict(cls, dictionary):
        """Load data from dictionary"""
        return cls(
            array=dictionary['array'],
            names=dictionary['names'],
        )

    def to_dict(self, iteration=None):
        """Convert data to dictionary"""
        return {
            'array': to_array(self.array, iteration),
            'names': self.names,
        }


class SensorsData(SensorsDataCy):
    """SensorsData"""

    @classmethod
    def from_dict(cls, dictionary):
        """Load data from dictionary"""
        return cls(
            contacts=ContactsArray.from_dict(
                dictionary['contacts']
            ),
            proprioception=ProprioceptionArray.from_dict(
                dictionary['proprioception']
            ),
            gps=GpsArray.from_dict(
                dictionary['gps']
            ),
            hydrodynamics=HydrodynamicsArray.from_dict(
                dictionary['hydrodynamics']
            ),
        )

    def to_dict(self, iteration=None):
        """Convert data to dictionary"""
        return {
            name: data.to_dict(iteration)
            for name, data in [
                ['contacts', self.contacts],
                ['proprioception', self.proprioception],
                ['gps', self.gps],
                ['hydrodynamics', self.hydrodynamics],
            ]
        }

    def plot(self, times):
        """Plot"""
        self.contacts.plot(times)
        self.proprioception.plot(times)
        self.gps.plot(times)
        self.hydrodynamics.plot(times)


class ContactsArray(SensorData, ContactsArrayCy):
    """Sensor array"""

    @classmethod
    def from_names(cls, names, n_iterations):
        """From names"""
        n_sensors = len(names)
        array = np.zeros([n_iterations, n_sensors, 12], dtype=NPDTYPE)
        return cls(array, names)

    @classmethod
    def from_parameters(cls, n_iterations, n_contacts, names):
        """From parameters"""
        return cls(np.zeros([n_iterations, n_contacts, 12]), names)

    @classmethod
    def from_size(cls, n_contacts, n_iterations, names):
        """From size"""
        contacts = np.zeros([n_iterations, n_contacts, 12], dtype=NPDTYPE)
        return cls(contacts, names)

    def reaction(self, iteration, sensor_i):
        """Reaction force"""
        return self.array[iteration, sensor_i, 0:3]

    def reaction_all(self, sensor_i):
        """Reaction force"""
        return self.array[:, sensor_i, 0:3]

    def friction(self, iteration, sensor_i):
        """Friction force"""
        return self.array[iteration, sensor_i, 3:6]

    def friction_all(self, sensor_i):
        """Friction force"""
        return self.array[:, sensor_i, 3:6]

    def total(self, iteration, sensor_i):
        """Total force"""
        return self.array[iteration, sensor_i, 6:9]

    def total_all(self, sensor_i):
        """Total force"""
        return self.array[:, sensor_i, 6:9]

    def position(self, iteration, sensor_i):
        """Position"""
        return self.array[iteration, sensor_i, 9:12]

    def position_all(self, sensor_i):
        """Positions"""
        return self.array[:, sensor_i, 9:12]

    def plot(self, times):
        """Plot"""
        self.plot_ground_reaction_forces(times)
        self.plot_ground_reaction_forces_all(times)
        self.plot_friction_forces(times)
        for ori in range(3):
            self.plot_friction_forces_ori(times, ori=ori)
        self.plot_total_forces(times)

    def plot_ground_reaction_forces(self, times):
        """Plot ground reaction forces"""
        plt.figure('Ground reaction forces')
        for sensor_i in range(self.size(1)):
            data = np.asarray(self.reaction_all(sensor_i))
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1)[:len(times)],
                label='Leg_{}'.format(sensor_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Forces [N]')
        plt.grid(True)

    def plot_ground_reaction_forces_all(self, times):
        """Plot ground reaction forces"""
        axis = ['x', 'y', 'z']
        for sensor_i, name in enumerate(self.names):
            data = np.asarray(self.reaction_all(sensor_i))
            plt.figure('Ground reaction forces {}'.format(name))
            for direction_i, direction in enumerate(['x', 'y', 'z']):
                plt.plot(
                    times,
                    data[:len(times), direction_i],
                    label='{}_{}'.format(name, direction)
                )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Forces [N]')
        plt.grid(True)

    def plot_friction_forces(self, times):
        """Plot friction forces"""
        plt.figure('Friction forces')
        for sensor_i in range(self.size(1)):
            data = np.asarray(self.friction_all(sensor_i))
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1)[:len(times)],
                label='Leg_{}'.format(sensor_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Forces [N]')
        plt.grid(True)

    def plot_friction_forces_ori(self, times, ori):
        """Plot friction forces"""
        plt.figure('Friction forces (ori={})'.format(ori))
        for sensor_i in range(self.size(1)):
            data = np.asarray(self.friction_all(sensor_i))
            plt.plot(
                times,
                data[:len(times), ori],
                label='Leg_{}'.format(sensor_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Forces [N]')
        plt.grid(True)

    def plot_total_forces(self, times):
        """Plot contact forces"""
        plt.figure('Contact total forces')
        for sensor_i in range(self.size(1)):
            data = np.asarray(self.total_all(sensor_i))
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1)[:len(times)],
                label='Leg_{}'.format(sensor_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Forces [N]')
        plt.grid(True)


class ProprioceptionArray(SensorData, ProprioceptionArrayCy):
    """Proprioception array"""

    @classmethod
    def from_names(cls, names, n_iterations):
        """From names"""
        n_sensors = len(names)
        array = np.zeros([n_iterations, n_sensors, 12], dtype=NPDTYPE)
        return cls(array, names)

    @classmethod
    def from_size(cls, n_joints, n_iterations, names):
        """From size"""
        proprioception = np.zeros([n_iterations, n_joints, 12], dtype=NPDTYPE)
        return cls(proprioception, names)

    @classmethod
    def from_parameters(cls, n_iterations, n_joints, names):
        """From parameters"""
        return cls(np.zeros([n_iterations, n_joints, 12]), names)

    def position(self, iteration, joint_i):
        """Joint position"""
        return self.array[iteration, joint_i, 0]

    def positions(self, iteration):
        """Joints positions"""
        return self.array[iteration, :, 0]

    def positions_all(self):
        """Joints positions"""
        return self.array[:, :, 0]

    def velocity(self, iteration, joint_i):
        """Joint velocity"""
        return self.array[iteration, joint_i, 1]

    def velocities(self, iteration):
        """Joints velocities"""
        return self.array[iteration, :, 1]

    def velocities_all(self):
        """Joints velocities"""
        return self.array[:, :, 1]

    def force(self, iteration, joint_i):
        """Joint force"""
        return self.array[iteration, joint_i, 2:5]

    def forces_all(self):
        """Joints forces"""
        return self.array[:, :, 2:5]

    def torque(self, iteration, joint_i):
        """Joint torque"""
        return self.array[iteration, joint_i, 5:8]

    def torques_all(self):
        """Joints torques"""
        return self.array[:, :, 5:8]

    def motor_torque(self, iteration, joint_i):
        """Joint velocity"""
        return self.array[iteration, joint_i, 8]

    def motor_torques(self):
        """Joint velocity"""
        return self.array[:, :, 8]

    def active(self, iteration, joint_i):
        """Active torque"""
        return self.array[iteration, joint_i, 9]

    def active_torques(self):
        """Active torques"""
        return self.array[:, :, 9]

    def spring(self, iteration, joint_i):
        """Passive spring torque"""
        return self.array[iteration, joint_i, 10]

    def spring_torques(self):
        """Spring torques"""
        return self.array[:, :, 10]

    def damping(self, iteration, joint_i):
        """passive damping torque"""
        return self.array[iteration, joint_i, 11]

    def damping_torques(self):
        """Damping torques"""
        return self.array[:, :, 11]

    def plot(self, times):
        """Plot"""
        self.plot_positions(times)
        self.plot_velocities(times)
        self.plot_forces(times)
        self.plot_torques(times)
        self.plot_motor_torques(times)
        self.plot_active_torques(times)
        self.plot_spring_torques(times)
        self.plot_damping_torques(times)

    def plot_positions(self, times):
        """Plot ground reaction forces"""
        plt.figure('Joints positions')
        for joint_i in range(self.size(1)):
            plt.plot(
                times,
                np.asarray(self.positions_all())[:len(times), joint_i],
                label='Joint_{}'.format(joint_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Joint position [rad]')
        plt.grid(True)

    def plot_velocities(self, times):
        """Plot ground reaction forces"""
        plt.figure('Joints velocities')
        for joint_i in range(self.size(1)):
            plt.plot(
                times,
                np.asarray(self.velocities_all())[:len(times), joint_i],
                label='Joint_{}'.format(joint_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Joint velocity [rad/s]')
        plt.grid(True)

    def plot_forces(self, times):
        """Plot ground reaction forces"""
        plt.figure('Joints forces')
        for joint_i in range(self.size(1)):
            data = np.linalg.norm(np.asarray(self.forces_all()), axis=-1)
            plt.plot(
                times,
                data[:len(times), joint_i],
                label='Joint_{}'.format(joint_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Joint force [N]')
        plt.grid(True)

    def plot_torques(self, times):
        """Plot ground reaction torques"""
        plt.figure('Joints torques')
        for joint_i in range(self.size(1)):
            data = np.linalg.norm(np.asarray(self.torques_all()), axis=-1)
            plt.plot(
                times,
                data[:len(times), joint_i],
                label='Joint_{}'.format(joint_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Joint torque [Nm]')
        plt.grid(True)

    def plot_motor_torques(self, times):
        """Plot ground reaction forces"""
        plt.figure('Joints motor torques')
        for joint_i in range(self.size(1)):
            plt.plot(
                times,
                np.asarray(self.motor_torques())[:len(times), joint_i],
                label='Joint_{}'.format(joint_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Joint torque [Nm]')
        plt.grid(True)

    def plot_active_torques(self, times):
        """Plot joints active torques"""
        plt.figure('Joints active torques')
        for joint_i in range(self.size(1)):
            plt.plot(
                times,
                np.asarray(self.active_torques())[:len(times), joint_i],
                label='Joint_{}'.format(joint_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Joint torque [Nm]')
        plt.grid(True)

    def plot_spring_torques(self, times):
        """Plot joints spring torques"""
        plt.figure('Joints spring torques')
        for joint_i in range(self.size(1)):
            plt.plot(
                times,
                np.asarray(self.spring_torques())[:len(times), joint_i],
                label='Joint_{}'.format(joint_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Joint torque [Nm]')
        plt.grid(True)

    def plot_damping_torques(self, times):
        """Plot joints damping torques"""
        plt.figure('Joints damping torques')
        for joint_i in range(self.size(1)):
            plt.plot(
                times,
                np.asarray(self.damping_torques())[:len(times), joint_i],
                label='Joint_{}'.format(joint_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Joint torque [Nm]')
        plt.grid(True)


class GpsArray(SensorData, GpsArrayCy):
    """Gps array"""

    @classmethod
    def from_names(cls, names, n_iterations):
        """From names"""
        n_sensors = len(names)
        array = np.zeros([n_iterations, n_sensors, 20], dtype=NPDTYPE)
        return cls(array, names)

    @classmethod
    def from_size(cls, n_links, n_iterations, names):
        """From size"""
        gps = np.zeros([n_iterations, n_links, 20], dtype=NPDTYPE)
        return cls(gps, names)

    @classmethod
    def from_parameters(cls, n_iterations, n_links, names):
        """From parameters"""
        return cls(np.zeros([n_iterations, n_links, 20]), names)

    def com_position(self, iteration, link_i):
        """CoM position of a link"""
        return self.array[iteration, link_i, 0:3]

    def com_orientation(self, iteration, link_i):
        """CoM orientation of a link"""
        return self.array[iteration, link_i, 3:7]

    def urdf_position(self, iteration, link_i):
        """URDF position of a link"""
        return self.array[iteration, link_i, 7:10]

    def urdf_positions(self):
        """URDF position of a link"""
        return self.array[:, :, 7:10]

    def urdf_orientation(self, iteration, link_i):
        """URDF orientation of a link"""
        return self.array[iteration, link_i, 10:14]

    def urdf_orientations(self):
        """URDF orientation of a link"""
        return self.array[:, :, 10:14]

    def com_lin_velocity(self, iteration, link_i):
        """CoM linear velocity of a link"""
        return self.array[iteration, link_i, 14:17]

    def com_lin_velocities(self):
        """CoM linear velocities"""
        return self.array[:, :, 14:17]

    def com_ang_velocity(self, iteration, link_i):
        """CoM angular velocity of a link"""
        return self.array[iteration, link_i, 17:20]

    def plot(self, times):
        """Plot"""
        self.plot_base_position(times, xaxis=0, yaxis=1)
        self.plot_base_velocity(times)

    def plot_base_position(self, times, xaxis=0, yaxis=1):
        """Plot"""
        plt.figure('Links position')
        for link_i in range(self.size(1)):
            data = np.asarray(self.urdf_positions())[:len(times), link_i]
            plt.plot(
                data[:, xaxis],
                data[:, yaxis],
                label='Link_{}'.format(link_i)
            )
        plt.legend()
        plt.xlabel('Position [m]')
        plt.ylabel('Position [m]')
        plt.axis('equal')
        plt.grid(True)

    def plot_base_velocity(self, times):
        """Plot"""
        plt.figure('Links velocities')
        for link_i in range(self.size(1)):
            data = np.asarray(self.com_lin_velocities())[:len(times), link_i]
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1),
                label='Link_{}'.format(link_i)
            )
        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.grid(True)


class HydrodynamicsArray(SensorData, HydrodynamicsArrayCy):
    """Hydrodynamics array"""

    @classmethod
    def from_names(cls, names, n_iterations):
        """From names"""
        n_sensors = len(names)
        array = np.zeros([n_iterations, n_sensors, 6], dtype=NPDTYPE)
        return cls(array, names)

    @classmethod
    def from_size(cls, n_links, n_iterations, names):
        """From size"""
        hydrodynamics = np.zeros([n_iterations, n_links, 6], dtype=NPDTYPE)
        return cls(hydrodynamics, names)

    @classmethod
    def from_parameters(cls, n_iterations, n_links, names):
        """From parameters"""
        return cls(np.zeros([n_iterations, n_links, 6]), names)

    def force(self, iteration, sensor_i):
        """Force"""
        return self.array[iteration, sensor_i, 0:3]

    def forces(self):
        """Forces"""
        return self.array[:, :, 0:3]

    def set_force(self, iteration, sensor_i, force):
        """Set force"""
        self.array[iteration, sensor_i, 0:3] = force

    def torque(self, iteration, sensor_i):
        """Torque"""
        return self.array[iteration, sensor_i, 3:6]

    def torques(self):
        """Torques"""
        return self.array[:, :, 3:6]

    def set_torque(self, iteration, sensor_i, torque):
        """Set torque"""
        self.array[iteration, sensor_i, 3:6] = torque

    def plot(self, times):
        """Plot"""
        self.plot_forces(times)
        self.plot_torques(times)

    def plot_forces(self, times):
        """Plot"""
        plt.figure('Hydrodynamic forces')
        for link_i in range(self.size(1)):
            data = np.asarray(self.forces())[:len(times), link_i]
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1),
                label='Link_{}'.format(link_i)
            )
        plt.xlabel('Time [s]')
        plt.ylabel('Forces [N]')
        plt.grid(True)

    def plot_torques(self, times):
        """Plot"""
        plt.figure('Hydrodynamic torques')
        for link_i in range(self.size(1)):
            data = np.asarray(self.torques())[:len(times), link_i]
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1),
                label='Link_{}'.format(link_i)
            )
        plt.xlabel('Time [s]')
        plt.ylabel('Torques [Nm]')
        plt.grid(True)
