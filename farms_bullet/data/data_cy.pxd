"""Animat data"""

include 'types.pxd'
from .array cimport DoubleArray3D


cdef class SensorsDataCy:
    """SensorsData"""
    cdef public ContactsArrayCy contacts
    cdef public ProprioceptionArrayCy proprioception
    cdef public GpsArrayCy gps
    cdef public HydrodynamicsArrayCy hydrodynamics


cdef enum:

    # Contacts
    CONTACT_REACTION_X = 0
    CONTACT_REACTION_Y = 1
    CONTACT_REACTION_Z = 2
    CONTACT_FRICTION_X = 3
    CONTACT_FRICTION_Y = 4
    CONTACT_FRICTION_Z = 5
    CONTACT_TOTAL_X = 6
    CONTACT_TOTAL_Y = 7
    CONTACT_TOTAL_Z = 8

    # Proprioception
    PROPRIOCEPTION_POSITION = 0
    PROPRIOCEPTION_VELOCITY = 1
    PROPRIOCEPTION_FORCE_X = 2
    PROPRIOCEPTION_FORCE_Y = 3
    PROPRIOCEPTION_FORCE_Z = 4
    PROPRIOCEPTION_TORQUE_X = 5
    PROPRIOCEPTION_TORQUE_Y = 6
    PROPRIOCEPTION_TORQUE_Z = 7
    PROPRIOCEPTION_TORQUE = 8
    PROPRIOCEPTION_TORQUE_ACTIVE = 9
    PROPRIOCEPTION_TORQUE_STIFFNESS = 10
    PROPRIOCEPTION_TORQUE_DAMPING = 11

    # GPS
    GPS_COM_POSITION_X = 0
    GPS_COM_POSITION_Y = 1
    GPS_COM_POSITION_Z = 2
    GPS_COM_ORIENTATION_X = 3
    GPS_COM_ORIENTATION_Y = 4
    GPS_COM_ORIENTATION_Z = 5
    GPS_COM_ORIENTATION_W = 6
    GPS_URDF_POSITION_X = 7
    GPS_URDF_POSITION_Y = 8
    GPS_URDF_POSITION_Z = 9
    GPS_URDF_ORIENTATION_X = 10
    GPS_URDF_ORIENTATION_Y = 11
    GPS_URDF_ORIENTATION_Z = 12
    GPS_URDF_ORIENTATION_W = 13
    GPS_COM_VELOCITY_LIN_X = 14
    GPS_COM_VELOCITY_LIN_Y = 15
    GPS_COM_VELOCITY_LIN_Z = 16
    GPS_COM_VELOCITY_ANG_X = 17
    GPS_COM_VELOCITY_ANG_Y = 18
    GPS_COM_VELOCITY_ANG_Z = 19



cdef class ContactsArrayCy(DoubleArray3D):
    """Sensor array"""

    cdef inline DTYPEv1 c_all(self, unsigned iteration, unsigned int index) nogil:
        """Reaction"""
        return self.array[iteration, index, :]

    cdef inline DTYPEv1 c_reaction(self, unsigned iteration, unsigned int index) nogil:
        """Reaction"""
        return self.array[iteration, index, CONTACT_REACTION_X:CONTACT_REACTION_Z+1]

    cdef inline DTYPE c_reaction_x(self, unsigned iteration, unsigned int index) nogil:
        """Reaction x"""
        return self.array[iteration, index, CONTACT_REACTION_X]

    cdef inline DTYPE c_reaction_y(self, unsigned iteration, unsigned int index) nogil:
        """Reaction y"""
        return self.array[iteration, index, CONTACT_REACTION_Y]

    cdef inline DTYPE c_reaction_z(self, unsigned iteration, unsigned int index) nogil:
        """Reaction z"""
        return self.array[iteration, index, CONTACT_REACTION_Z]

    cdef inline DTYPEv1 c_friction(self, unsigned iteration, unsigned int index) nogil:
        """Friction"""
        return self.array[iteration, index, CONTACT_FRICTION_X:CONTACT_FRICTION_Z+1]

    cdef inline DTYPE c_friction_x(self, unsigned iteration, unsigned int index) nogil:
        """Friction x"""
        return self.array[iteration, index, CONTACT_FRICTION_X]

    cdef inline DTYPE c_friction_y(self, unsigned iteration, unsigned int index) nogil:
        """Friction y"""
        return self.array[iteration, index, CONTACT_FRICTION_Y]

    cdef inline DTYPE c_friction_z(self, unsigned iteration, unsigned int index) nogil:
        """Friction z"""
        return self.array[iteration, index, CONTACT_FRICTION_Z]

    cdef inline DTYPEv1 c_total(self, unsigned iteration, unsigned int index) nogil:
        """Total"""
        return self.array[iteration, index, CONTACT_TOTAL_X:CONTACT_TOTAL_Z+1]

    cdef inline DTYPE c_total_x(self, unsigned iteration, unsigned int index) nogil:
        """Total x"""
        return self.array[iteration, index, CONTACT_TOTAL_X]

    cdef inline DTYPE c_total_y(self, unsigned iteration, unsigned int index) nogil:
        """Total y"""
        return self.array[iteration, index, CONTACT_TOTAL_Y]

    cdef inline DTYPE c_total_z(self, unsigned iteration, unsigned int index) nogil:
        """Total z"""
        return self.array[iteration, index, CONTACT_TOTAL_Z]


cdef class ProprioceptionArrayCy(DoubleArray3D):
    """Proprioception array"""

    cdef inline DTYPE position_cy(self, unsigned int iteration, unsigned int joint_i):
        """Joint position"""
        return self.array[iteration, joint_i, PROPRIOCEPTION_POSITION]

    cdef inline DTYPEv1 positions_cy(self, unsigned int iteration):
        """Joints positions"""
        return self.array[iteration, :, PROPRIOCEPTION_POSITION]

    cdef inline DTYPEv2 positions_all_cy(self):
        """Joints positions"""
        return self.array[:, :, PROPRIOCEPTION_POSITION]

    cdef inline DTYPE velocity_cy(self, unsigned int iteration, unsigned int joint_i):
        """Joint velocity"""
        return self.array[iteration, joint_i, PROPRIOCEPTION_VELOCITY]

    cdef inline DTYPEv1 velocities_cy(self, unsigned int iteration):
        """Joints velocities"""
        return self.array[iteration, :, PROPRIOCEPTION_VELOCITY]

    cdef inline DTYPEv2 velocities_all_cy(self):
        """Joints velocities"""
        return self.array[:, :, PROPRIOCEPTION_VELOCITY]

    cdef inline DTYPE motor_torque_cy(self, unsigned int iteration, unsigned int joint_i):
        """Joint velocity"""
        return self.array[iteration, joint_i, PROPRIOCEPTION_TORQUE]

    cdef inline DTYPEv2 motor_torques_cy(self):
        """Joint velocity"""
        return self.array[:, :, PROPRIOCEPTION_TORQUE]

    cdef inline DTYPEv1 force_cy(self, unsigned int iteration, unsigned int joint_i):
        """Joint force"""
        return self.array[iteration, joint_i, PROPRIOCEPTION_FORCE_X:PROPRIOCEPTION_FORCE_Z+1]

    cdef inline DTYPEv3 forces_all_cy(self):
        """Joints forces"""
        return self.array[:, :, PROPRIOCEPTION_FORCE_X:PROPRIOCEPTION_FORCE_Z+1]

    cdef inline DTYPEv1 torque_cy(self, unsigned int iteration, unsigned int joint_i):
        """Joint torque"""
        return self.array[iteration, joint_i, PROPRIOCEPTION_TORQUE_X:PROPRIOCEPTION_TORQUE_Z+1]

    cdef inline DTYPEv3 torques_all_cy(self):
        """Joints torques"""
        return self.array[:, :, PROPRIOCEPTION_TORQUE_X:PROPRIOCEPTION_TORQUE_Z+1]

    cdef inline DTYPE active_cy(self, unsigned int iteration, unsigned int joint_i):
        """Active torque"""
        return self.array[iteration, joint_i, PROPRIOCEPTION_TORQUE_ACTIVE]

    cdef inline DTYPEv2 active_torques_cy(self):
        """Active torques"""
        return self.array[:, :, PROPRIOCEPTION_TORQUE_ACTIVE]

    cdef inline DTYPE spring_cy(self, unsigned int iteration, unsigned int joint_i):
        """Passive spring torque"""
        return self.array[iteration, joint_i, PROPRIOCEPTION_TORQUE_STIFFNESS]

    cdef inline DTYPEv2 spring_torques_cy(self):
        """Spring torques"""
        return self.array[:, :, PROPRIOCEPTION_TORQUE_STIFFNESS]

    cdef inline DTYPE damping_cy(self, unsigned int iteration, unsigned int joint_i):
        """passive damping torque"""
        return self.array[iteration, joint_i, PROPRIOCEPTION_TORQUE_DAMPING]

    cdef inline DTYPEv2 damping_torques_cy(self):
        """Damping torques"""
        return self.array[:, :, PROPRIOCEPTION_TORQUE_DAMPING]


cdef class GpsArrayCy(DoubleArray3D):
    """Gps array"""

    cdef inline DTYPEv1 com_position_cy(self, unsigned int iteration, unsigned int link_i):
        """CoM position of a link"""
        return self.array[iteration, link_i, GPS_COM_POSITION_X:GPS_COM_POSITION_Z+1]

    cdef inline DTYPEv1 com_orientation_cy(self, unsigned int iteration, unsigned int link_i):
        """CoM orientation of a link"""
        return self.array[iteration, link_i, GPS_COM_ORIENTATION_X:GPS_COM_ORIENTATION_W+1]

    cdef inline DTYPEv1 urdf_position_cy(self, unsigned int iteration, unsigned int link_i):
        """URDF position of a link"""
        return self.array[iteration, link_i, GPS_URDF_POSITION_X:GPS_URDF_POSITION_Z+1]

    cdef inline DTYPEv3 urdf_positions_cy(self):
        """URDF position of a link"""
        return self.array[:, :, GPS_URDF_POSITION_X:GPS_URDF_POSITION_Z+1]

    cdef inline DTYPEv1 urdf_orientation_cy(self, unsigned int iteration, unsigned int link_i):
        """URDF orientation of a link"""
        return self.array[iteration, link_i, GPS_URDF_ORIENTATION_X:GPS_URDF_ORIENTATION_W+1]

    cdef inline DTYPEv1 com_lin_velocity_cy(self, unsigned int iteration, unsigned int link_i):
        """CoM linear velocity of a link"""
        return self.array[iteration, link_i, GPS_COM_VELOCITY_LIN_X:GPS_COM_VELOCITY_LIN_Z+1]

    cdef inline DTYPEv3 com_lin_velocities_cy(self):
        """CoM linear velocities"""
        return self.array[:, :, GPS_COM_VELOCITY_LIN_X:GPS_COM_VELOCITY_LIN_Z+1]

    cdef inline DTYPEv1 com_ang_velocity_cy(self, unsigned int iteration, unsigned int link_i):
        """CoM angular velocity of a link"""
        return self.array[iteration, link_i, GPS_COM_VELOCITY_ANG_X:GPS_COM_VELOCITY_ANG_Z+1]


cdef class HydrodynamicsArrayCy(DoubleArray3D):
    """Hydrodynamics array"""

    cdef inline DTYPE c_force_x(self, unsigned iteration, unsigned int index) nogil:
        """Force x"""
        return self.array[iteration, index, 0]

    cdef inline DTYPE c_force_y(self, unsigned iteration, unsigned int index) nogil:
        """Force y"""
        return self.array[iteration, index, 1]

    cdef inline DTYPE c_force_z(self, unsigned iteration, unsigned int index) nogil:
        """Force z"""
        return self.array[iteration, index, 2]

    cdef inline DTYPE c_torque_x(self, unsigned iteration, unsigned int index) nogil:
        """Torque x"""
        return self.array[iteration, index, 0]

    cdef inline DTYPE c_torque_y(self, unsigned iteration, unsigned int index) nogil:
        """Torque y"""
        return self.array[iteration, index, 1]

    cdef inline DTYPE c_torque_z(self, unsigned iteration, unsigned int index) nogil:
        """Torque z"""
        return self.array[iteration, index, 2]
