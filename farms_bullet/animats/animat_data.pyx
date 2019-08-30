"""Animat data"""

import numpy as np
cimport numpy as np
import matplotlib.pyplot as plt


cdef class AnimatData:
    """Network parameter"""

    def __init__(self, state=None, network=None, joints=None, sensors=None):
        super(AnimatData, self).__init__()
        self.state = state
        self.network = network
        self.joints = joints
        self.sensors = sensors
        self.iteration = 0

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
        plt.show()


cdef class NetworkParameters:
    """Network parameter"""

    def __init__(
            self,
            oscillators,
            connectivity,
            contacts_connectivity,
            hydro_connectivity
    ):
        super(NetworkParameters, self).__init__()
        self.oscillators = oscillators
        self.connectivity = connectivity
        self.contacts_connectivity = contacts_connectivity
        self.hydro_connectivity = hydro_connectivity

    def log(self, times, folder, extension):
        """Log"""
        for data, name in [
                [self.oscillators, "oscillators"],
                [self.connectivity, "connectivity"],
                [self.contacts_connectivity, "contacts_connectivity"],
                [self.hydro_connectivity, "hydro_connectivity"]
        ]:
            data.log(times, folder, name, extension)


cdef class OscillatorNetworkState(NetworkArray3D):
    """Network state"""

    def __init__(self, state, n_oscillators, iteration=0):
        super(OscillatorNetworkState, self).__init__(state)
        self.n_oscillators = n_oscillators
        self._iteration = iteration

    @classmethod
    def from_solver(cls, solver, n_oscillators):
        """From solver"""
        return cls(solver.state, n_oscillators, solver.iteration)

    def phases(self, unsigned int iteration):
        """Phases"""
        return self.array[iteration, 0, :self.n_oscillators]

    def phases_all(self):
        """Phases"""
        return self.array[:, 0, :self.n_oscillators]

    def amplitudes(self, unsigned int iteration):
        """Amplitudes"""
        return self.array[iteration, 0, self.n_oscillators:]

    def amplitudes_all(self):
        """Phases"""
        return self.array[:, 0, self.n_oscillators:]

    def dphases(self, unsigned int iteration):
        """Phases derivative"""
        return self.array[iteration, 1, :self.n_oscillators]

    def damplitudes(self, unsigned int iteration):
        """Amplitudes derivative"""
        return self.array[iteration, 1, self.n_oscillators:]

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


cdef class OscillatorArray(NetworkArray2D):
    """Oscillator array"""

    @classmethod
    def from_parameters(cls, freqs, rates, amplitudes):
        """From each parameter"""
        return cls(np.array([freqs, rates, amplitudes]))

    @property
    def freqs(self):
        """Frequencies"""
        return self.array[0]

    @freqs.setter
    def freqs(self, value):
        """Frequencies"""
        self.array[0, :] = value

    @property
    def amplitudes_rates(self):
        """Amplitudes rates"""
        return self.array[1]

    @property
    def amplitudes_desired(self):
        """Amplitudes desired"""
        return self.array[2]

    @amplitudes_desired.setter
    def amplitudes_desired(self, value):
        """Amplitudes desired"""
        self.array[2, :] = value


cdef class ConnectivityArray(NetworkArray2D):
    """Connectivity array"""

    @classmethod
    def from_parameters(cls, connections, weights, desired_phases):
        """From each parameter"""
        return cls(np.stack([connections, weights, desired_phases], axis=1))

    @property
    def connections(self):
        """Connections"""
        return self.array[:][0, 1]

    @property
    def weights(self):
        """Weights"""
        return self.array[:][2]

    @property
    def desired_phases(self):
        """Weights"""
        return self.array[:][3]


cdef class JointsArray(NetworkArray2D):
    """Oscillator array"""

    @classmethod
    def from_parameters(cls, offsets, rates):
        """From each parameter"""
        return cls(np.array([offsets, rates]))

    @property
    def offsets(self):
        """Joints angles offsets"""
        return self.array[0]

    @property
    def rates(self):
        """Joints angles offsets rates"""
        return self.array[1]

    def set_body_offset(self, value, n_body_joints=11):
        """Body offset"""
        self.array[0, :n_body_joints] = value

    def set_legs_offset(self, value, n_body_joints=11):
        """Legs offset"""
        self.array[0, n_body_joints:] = value


cdef class Sensors:
    """Sensors"""

    def __init__(
            self,
            ContactsArray contacts=None,
            ProprioceptionArray proprioception=None,
            GpsArray gps=None,
            HydrodynamicsArray hydrodynamics=None
    ):
        super(Sensors, self).__init__()
        self.contacts = contacts
        self.proprioception = proprioception
        self.gps = gps
        self.hydrodynamics = hydrodynamics

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


cdef class ContactsArray(NetworkArray3D):
    """Sensor array"""

    @classmethod
    def from_parameters(cls, n_iterations, n_contacts):
        """From parameters"""
        return cls(np.zeros([n_iterations, n_contacts, 9]))

    cpdef double[:] reaction(self, unsigned int iteration, unsigned int sensor_i):
        """Reaction force"""
        return self.array[iteration, sensor_i, 0:3]

    cpdef double[:, :] reaction_all(self, unsigned int sensor_i):
        """Reaction force"""
        return self.array[:, sensor_i, 0:3]

    cpdef double[:] friction(self, unsigned int iteration, unsigned int sensor_i):
        """Friction force"""
        return self.array[iteration, sensor_i, 3:6]

    cpdef double[:, :] friction_all(self, unsigned int sensor_i):
        """Friction force"""
        return self.array[:, sensor_i, 3:6]

    cpdef double[:] total(self, unsigned int iteration, unsigned int sensor_i):
        """Total force"""
        return self.array[iteration, sensor_i, 6:9]

    cpdef double[:, :] total_all(self, unsigned int sensor_i):
        """Total force"""
        return self.array[:, sensor_i, 6:9]

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


cdef class ProprioceptionArray(NetworkArray3D):
    """Proprioception array"""

    @classmethod
    def from_parameters(cls, n_iterations, n_joints):
        """From parameters"""
        return cls(np.zeros([n_iterations, n_joints, 9]))

    cpdef double position(self, unsigned int iteration, unsigned int joint_i):
        """Joint position"""
        return self.array[iteration, joint_i, 0]

    cpdef double[:] positions(self, unsigned int iteration):
        """Joints positions"""
        return self.array[iteration, :, 0]

    cpdef double[:, :] positions_all(self):
        """Joints positions"""
        return self.array[:, :, 0]

    cpdef double velocity(self, unsigned int iteration, unsigned int joint_i):
        """Joint velocity"""
        return self.array[iteration, joint_i, 1]

    cpdef double[:] velocities(self, unsigned int iteration):
        """Joints velocities"""
        return self.array[iteration, :, 1]

    cpdef double[:, :] velocities_all(self):
        """Joints velocities"""
        return self.array[:, :, 1]

    cpdef double[:] force(self, unsigned int iteration, unsigned int joint_i):
        """Joint force"""
        return self.array[iteration, joint_i, 2:5]

    cpdef double[:, :, :] forces_all(self):
        """Joints forces"""
        return self.array[:, :, 2:5]

    cpdef double[:] torque(self, unsigned int iteration, unsigned int joint_i):
        """Joint torque"""
        return self.array[iteration, joint_i, 5:8]

    cpdef double[:, :, :] torques_all(self):
        """Joints torques"""
        return self.array[:, :, 5:8]

    cpdef double motor_torque(self, unsigned int iteration, unsigned int joint_i):
        """Joint velocity"""
        return self.array[iteration, joint_i, 8]

    cpdef double[:, :] motor_torques(self):
        """Joint velocity"""
        return self.array[:, :, 8]

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
        plt.ylabel("Joint torque [rad]")
        plt.grid(True)


cdef class GpsArray(NetworkArray3D):
    """Gps array"""

    @classmethod
    def from_parameters(cls, n_iterations, n_links):
        """From parameters"""
        return cls(np.zeros([n_iterations, n_links, 20]))

    cpdef double[:] com_position(self, unsigned int iteration, unsigned int link_i):
        """CoM position of a link"""
        return self.array[iteration, link_i, 0:3]

    cpdef double[:] com_orientation(self, unsigned int iteration, unsigned int link_i):
        """CoM orientation of a link"""
        return self.array[iteration, link_i, 3:7]

    cpdef double[:] urdf_position(self, unsigned int iteration, unsigned int link_i):
        """URDF position of a link"""
        return self.array[iteration, link_i, 7:10]

    cpdef double[:] urdf_orientation(self, unsigned int iteration, unsigned int link_i):
        """URDF orientation of a link"""
        return self.array[iteration, link_i, 10:14]

    cpdef double[:] com_lin_velocity(self, unsigned int iteration, unsigned int link_i):
        """CoM linear velocity of a link"""
        return self.array[iteration, link_i, 14:17]

    cpdef double[:] com_ang_velocity(self, unsigned int iteration, unsigned int link_i):
        """CoM angular velocity of a link"""
        return self.array[iteration, link_i, 17:20]

    def plot(self, times):
        """Plot"""
        pass


cdef class HydrodynamicsArray(NetworkArray3D):
    """Hydrodynamics array"""

    @classmethod
    def from_parameters(cls, n_iterations, n_links):
        """From parameters"""
        return cls(np.zeros([n_iterations, n_links, 6]))

    def plot(self, times):
        """Plot"""
        pass
