"""Cython sensors"""

import numpy as np
cimport numpy as np

import pybullet

from farms_data.sensors.data_cy cimport ContactsArrayCy
from dm_control.mujoco.wrapper.core import mjlib

from libc.math cimport sqrt


cdef double norm3d(double[3] vector) nogil:
    """Compute 3D norm"""
    return sqrt(vector[0]*vector[0] + vector[1]*vector[1] + vector[2]*vector[2])


cdef void store_forces(
    unsigned int iteration,
    unsigned int index,
    DTYPEv3 cdata,
    DTYPEv1 forcetorque,
    double[9] frame,
    double[3] pos,
    int sign,
) nogil:
    """Store forces"""
    cdef double norm
    cdef double[3] reaction, friction, friction1, friction2, total
    for i in range(3):
        reaction[i] = sign*forcetorque[i]*frame[0+i]
        friction1[i] = sign*forcetorque[i]*frame[3+i]
        friction2[i] = sign*forcetorque[i]*frame[6+i]
        friction[i] = friction1[i] + friction2[i]
        total[i] = reaction[i] + friction[i]
    cdata[iteration, index, CONTACT_REACTION_X] += reaction[0]
    cdata[iteration, index, CONTACT_REACTION_Y] += reaction[1]
    cdata[iteration, index, CONTACT_REACTION_Z] += reaction[2]
    cdata[iteration, index, CONTACT_FRICTION_X] += friction[0]
    cdata[iteration, index, CONTACT_FRICTION_Y] += friction[1]
    cdata[iteration, index, CONTACT_FRICTION_Z] += friction[2]
    cdata[iteration, index, CONTACT_TOTAL_X] += total[0]
    cdata[iteration, index, CONTACT_TOTAL_Y] += total[1]
    cdata[iteration, index, CONTACT_TOTAL_Z] += total[2]
    norm = norm3d(total)
    cdata[iteration, index, CONTACT_POSITION_X] += norm*pos[0]
    cdata[iteration, index, CONTACT_POSITION_Y] += norm*pos[1]
    cdata[iteration, index, CONTACT_POSITION_Z] += norm*pos[2]


cdef void normalize_forces_pos(
    unsigned int iteration,
    unsigned int index,
    DTYPEv3 cdata,
) nogil:
    """Normalize forces position"""
    cdef double[3] vector
    vector[0] = cdata[iteration, index, CONTACT_TOTAL_X]
    vector[1] = cdata[iteration, index, CONTACT_TOTAL_Y]
    vector[2] = cdata[iteration, index, CONTACT_TOTAL_Z]
    cdef double norm = norm3d(vector)
    if norm > 0:
        cdata[iteration, index, CONTACT_POSITION_X] /= norm
        cdata[iteration, index, CONTACT_POSITION_Y] /= norm
        cdata[iteration, index, CONTACT_POSITION_Z] /= norm


cdef cycontact2data(
    unsigned int iteration,
    unsigned int contact_i,
    unsigned int index,
    object model_ptr,
    object data_ptr,
    object contact,
    DTYPEv3 cdata,
    int sign,
):
    """Extract force"""
    cdef double[9] frame
    cdef double[3] pos
    cdef np.ndarray[double, ndim=1] forcetorque = np.empty(6, dtype=np.double)
    frame[:] = contact.frame
    pos[:] = contact.pos
    mjlib.mj_contactForce(model_ptr, data_ptr, contact_i, forcetorque)
    store_forces(
        iteration=iteration, index=index, cdata=cdata,
        forcetorque=forcetorque,
        frame=frame, pos=pos, sign=sign,
    )


cpdef cycontacts2data(
    object physics,
    unsigned int iteration,
    ContactsArrayCy data,
    dict geom2data,
    set geom_set
):
    """Contacts to data"""
    cdef unsigned int contact_i
    cdef DTYPEv3 cdata = data.array
    cdef object model_ptr = physics.model.ptr
    cdef object data_ptr = physics.data.ptr
    cdef object contacts = physics.data.contact
    for contact_i in range(len(physics.data.contact)):
        # Extract body index
        contact = contacts[contact_i]
        if contact.geom1 in geom_set:
            cycontact2data(
                iteration=iteration,
                contact_i=contact_i,
                index=geom2data[contact.geom1],
                model_ptr=model_ptr,
                data_ptr=data_ptr,
                contact=contact,
                cdata=cdata,
                sign=1,
            )
        if contact.geom2 in geom_set:
            cycontact2data(
                iteration=iteration,
                contact_i=contact_i,
                index=geom2data[contact.geom2],
                model_ptr=model_ptr,
                data_ptr=data_ptr,
                contact=contact,
                cdata=cdata,
                sign=-1,
            )
    for index in range(len(data.names)):
        normalize_forces_pos(
            iteration=iteration,
            index=index,
            cdata=cdata,
        )


cdef class Sensors(dict):
    """Sensors"""

    def add(self, new_dict):
        """Add sensors"""
        dict.update(self, new_dict)

    def update(self, iteration):
        """Update all sensors"""
        for sensor in self.values():
            sensor.update(iteration)


cdef class ContactsSensors(DoubleArray3D):
    """Model sensors"""

    def __init__(self, array, model_ids, model_links, meters=1, newtons=1):
        super(ContactsSensors, self).__init__(array)
        self.model_ids = np.array(model_ids, dtype=np.uintc)
        self.model_links = np.array(model_links, dtype=np.intc)
        self.imeters = 1./meters
        self.inewtons = 1./newtons

    cpdef tuple get_contacts(self, unsigned int model_id, int model_link):
        """Get contacts"""
        return pybullet.getContactPoints(
            bodyA=model_id,
            linkIndexA=model_link,
        )

    cpdef void update(self, unsigned int iteration):
        """Update sensors"""
        cdef int model_link
        cdef unsigned int sensor_i, model_id
        cdef double rx, ry, rz, fx, fy, fz, px, py, pz
        cdef double rx_tot, ry_tot, rz_tot, fx_tot, fy_tot, fz_tot
        cdef double inewtons = self.inewtons
        cdef double imeters = self.imeters
        cdef tuple contact
        for sensor_i, (model_id, model_link) in enumerate(
                zip(self.model_ids, self.model_links)
        ):
            px = 0
            py = 0
            pz = 0
            rx_tot = 0
            ry_tot = 0
            rz_tot = 0
            fx_tot = 0
            fy_tot = 0
            fz_tot = 0
            for contact in self.get_contacts(model_id, model_link):
                # Normal reaction
                rx = contact[9]*contact[7][0]*inewtons
                ry = contact[9]*contact[7][1]*inewtons
                rz = contact[9]*contact[7][2]*inewtons
                rx_tot += rx
                ry_tot += ry
                rz_tot += rz
                # Lateral friction dir 1 + Lateral friction dir 2
                fx = (contact[10]*contact[11][0] + contact[12]*contact[13][0])*inewtons
                fy = (contact[10]*contact[11][1] + contact[12]*contact[13][1])*inewtons
                fz = (contact[10]*contact[11][2] + contact[12]*contact[13][2])*inewtons
                fx_tot += fx
                fy_tot += fy
                fz_tot += fz
                # Position
                px += (rx+fx)*contact[5][0]*imeters
                py += (ry+fy)*contact[5][1]*imeters
                pz += (rz+fz)*contact[5][2]*imeters
            self.array[iteration, sensor_i, CONTACT_REACTION_X] = rx_tot
            self.array[iteration, sensor_i, CONTACT_REACTION_Y] = ry_tot
            self.array[iteration, sensor_i, CONTACT_REACTION_Z] = rz_tot
            self.array[iteration, sensor_i, CONTACT_FRICTION_X] = fx_tot
            self.array[iteration, sensor_i, CONTACT_FRICTION_Y] = fy_tot
            self.array[iteration, sensor_i, CONTACT_FRICTION_Z] = fz_tot
            self.array[iteration, sensor_i, CONTACT_TOTAL_X] = rx_tot + fx_tot
            self.array[iteration, sensor_i, CONTACT_TOTAL_Y] = ry_tot + fy_tot
            self.array[iteration, sensor_i, CONTACT_TOTAL_Z] = rz_tot + fz_tot
            if self.array[iteration, sensor_i, CONTACT_TOTAL_X] != 0:
                self.array[iteration, sensor_i, CONTACT_POSITION_X] = px/(rx_tot + fx_tot)
            if self.array[iteration, sensor_i, CONTACT_TOTAL_Y] != 0:
                self.array[iteration, sensor_i, CONTACT_POSITION_Y] = py/(ry_tot + fy_tot)
            if self.array[iteration, sensor_i, CONTACT_TOTAL_Z] != 0:
                self.array[iteration, sensor_i, CONTACT_POSITION_Z] = pz/(rz_tot + fz_tot)


cdef class JointsStatesSensor(DoubleArray3D):
    """Joint state sensor"""

    def __init__(self, array, model_id, joints, units, enable_ft=True):
        super(JointsStatesSensor, self).__init__(array)
        self.model_id = model_id
        self.joints_map = joints
        self.seconds = units.seconds
        self.inewtons = 1./units.newtons
        self.itorques = 1./units.torques
        if enable_ft:
            for joint in self.joints_map:
                pybullet.enableJointForceTorqueSensor(
                    self.model_id,
                    joint
                )

    cpdef tuple get_joints_states(self):
        """Get joints states"""
        return pybullet.getJointStates(self.model_id, self.joints_map)

    cpdef void update(self, unsigned int iteration):
        """Update sensor"""
        cdef unsigned int joint_i
        cdef double position, velocity, torque, fx, fy, fz, tx, ty, tz
        for (
                joint_i,
                (position, velocity, (fx, fy, fz, tx, ty, tz), torque),
        ) in enumerate(self.get_joints_states()):
            # Position
            self.array[iteration, joint_i, JOINT_POSITION] = position
            # Velocity
            self.array[iteration, joint_i, JOINT_VELOCITY] = velocity*self.seconds
            # Motor torque
            self.array[iteration, joint_i, JOINT_TORQUE] = torque*self.itorques
            # Forces
            self.array[iteration, joint_i, JOINT_FORCE_X] = fx*self.inewtons
            self.array[iteration, joint_i, JOINT_FORCE_Y] = fy*self.inewtons
            self.array[iteration, joint_i, JOINT_FORCE_Z] = fz*self.inewtons
            # Torques
            self.array[iteration, joint_i, JOINT_TORQUE_X] = tx*self.itorques
            self.array[iteration, joint_i, JOINT_TORQUE_Y] = ty*self.itorques
            self.array[iteration, joint_i, JOINT_TORQUE_Z] = tz*self.itorques


cdef class LinksStatesSensor(DoubleArray3D):
    """Links states sensor

    links is an array of size (N, 3) where the 3 values are:
    [
        link_name,  # Name of the link
        link_i,  # Index in table
        link_id  # Index in model
    ]
    """

    def __init__(self, array, model_id, links, units, transform_inv=None):
        super(LinksStatesSensor, self).__init__(array)
        self.model = model_id
        self.links = links
        self.imeters = 1./units.meters
        self.ivelocity = 1./units.velocity
        self.seconds = units.seconds
        if transform_inv is not None:
            self.transform_inv_pos, self.transform_inv_ori = transform_inv
        else:
            self.transform_inv_pos = None
            self.transform_inv_ori = None

    cpdef tuple get_base_link_state(self):
        """Get link states"""
        cdef tuple pos_com, ori_com, base_velocity
        cdef tuple transform_loc, transform_urdf
        pos_com, ori_com = pybullet.getBasePositionAndOrientation(self.model)
        if self.transform_inv_pos is None or self.transform_inv_ori is None:
            transform_loc = pybullet.getDynamicsInfo(self.model, -1)[3:5]
            self.transform_inv_pos, self.transform_inv_ori = pybullet.invertTransform(
                position=transform_loc[0],
                orientation=transform_loc[1],
            )
        transform_urdf = pybullet.multiplyTransforms(
            pos_com,
            ori_com,
            self.transform_inv_pos,
            self.transform_inv_ori,
        )
        base_velocity = pybullet.getBaseVelocity(self.model)
        return (
            pos_com,
            ori_com,
            transform_urdf[0],
            transform_urdf[1],
            base_velocity[0],
            base_velocity[1],
        )

    cpdef tuple get_children_links_states(self):
        """Get link states"""
        return pybullet.getLinkStates(
            self.model,
            self.links,
            computeLinkVelocity=1,
            computeForwardKinematics=1
        )

    cpdef void update(self, unsigned int iteration):
        """Update sensor"""
        cdef int link_id
        cdef unsigned int link_i
        cdef double pos_com[3]
        cdef double ori_com[4]
        cdef double pos_urdf[3]
        cdef double ori_urdf[4]
        cdef double lin_velocity[3]
        cdef double ang_velocity[3]
        states = self.get_children_links_states()
        for link_i, link_id in enumerate(self.links):
            # Collect data
            if link_id == -1:
                # Base link
                link_state = self.get_base_link_state()
                pos_com = link_state[0]
                ori_com = link_state[1]
                pos_urdf = link_state[2]
                ori_urdf = link_state[3]
                lin_velocity = link_state[4]
                ang_velocity = link_state[5]
            else:
                # Children links
                link_state = states[link_i]
                pos_com = link_state[0]  # Position of CoM
                ori_com = link_state[1]  # Orientation of CoM
                pos_urdf = link_state[4]  # Position of URDF frame
                ori_urdf = link_state[5]  # Orientation of URDF frame
                lin_velocity = link_state[6]  # Velocity of CoM
                ang_velocity = link_state[7]  # Angular velocity of CoM
            # Position of CoM
            self.array[iteration, link_i, LINK_COM_POSITION_X] = pos_com[0]*self.imeters
            self.array[iteration, link_i, LINK_COM_POSITION_Y] = pos_com[1]*self.imeters
            self.array[iteration, link_i, LINK_COM_POSITION_Z] = pos_com[2]*self.imeters
            # Orientation of CoM
            self.array[iteration, link_i, LINK_COM_ORIENTATION_X] = ori_com[0]
            self.array[iteration, link_i, LINK_COM_ORIENTATION_Y] = ori_com[1]
            self.array[iteration, link_i, LINK_COM_ORIENTATION_Z] = ori_com[2]
            self.array[iteration, link_i, LINK_COM_ORIENTATION_W] = ori_com[3]
            # Position of URDF frame
            self.array[iteration, link_i, LINK_URDF_POSITION_X] = pos_urdf[0]*self.imeters
            self.array[iteration, link_i, LINK_URDF_POSITION_Y] = pos_urdf[1]*self.imeters
            self.array[iteration, link_i, LINK_URDF_POSITION_Z] = pos_urdf[2]*self.imeters
            # Orientation of URDF frame
            self.array[iteration, link_i, LINK_URDF_ORIENTATION_X] = ori_urdf[0]
            self.array[iteration, link_i, LINK_URDF_ORIENTATION_Y] = ori_urdf[1]
            self.array[iteration, link_i, LINK_URDF_ORIENTATION_Z] = ori_urdf[2]
            self.array[iteration, link_i, LINK_URDF_ORIENTATION_W] = ori_urdf[3]
            # Velocity of CoM
            self.array[iteration, link_i, LINK_COM_VELOCITY_LIN_X] = lin_velocity[0]*self.ivelocity
            self.array[iteration, link_i, LINK_COM_VELOCITY_LIN_Y] = lin_velocity[1]*self.ivelocity
            self.array[iteration, link_i, LINK_COM_VELOCITY_LIN_Z] = lin_velocity[2]*self.ivelocity
            # Angular velocity of CoM
            self.array[iteration, link_i, LINK_COM_VELOCITY_ANG_X] = ang_velocity[0]*self.seconds
            self.array[iteration, link_i, LINK_COM_VELOCITY_ANG_Y] = ang_velocity[1]*self.seconds
            self.array[iteration, link_i, LINK_COM_VELOCITY_ANG_Z] = ang_velocity[2]*self.seconds
