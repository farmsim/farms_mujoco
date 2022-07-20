"""Cython sensors"""

import numpy as np
cimport numpy as np

from dm_control.mujoco.wrapper.core import mjlib

from libc.math cimport sqrt


cdef inline double norm3d(double[3] vector) nogil:
    """Compute 3D norm"""
    return sqrt(vector[0]*vector[0] + vector[1]*vector[1] + vector[2]*vector[2])


cdef double store_forces(
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
        reaction[i] = sign*forcetorque[0]*frame[0+i]
        friction1[i] = sign*forcetorque[1]*frame[3+i]
        friction2[i] = sign*forcetorque[2]*frame[6+i]
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
    return norm


cdef void cycontact2data(
    unsigned int iteration,
    unsigned int contact_i,
    unsigned int index,
    object model_ptr,
    object data_ptr,
    object contact,
    DTYPEv3 cdata,
    np.ndarray forcetorque,
    DTYPEv1 norm_sum,
    int sign,
):
    """Extract force"""
    cdef double[3] pos = contact.pos
    cdef double[9] frame = contact.frame
    mjlib.mj_contactForce(model_ptr, data_ptr, contact_i, forcetorque)
    norm_sum[index] += store_forces(
        iteration=iteration, index=index, cdata=cdata,
        forcetorque=forcetorque,
        frame=frame, pos=pos, sign=sign,
    )


cdef void normalize_forces_pos(
    unsigned int iteration,
    unsigned int index,
    DTYPEv3 cdata,
    DTYPEv1 norm_sum,
) nogil:
    """Normalize forces position"""
    if norm_sum[index] > 0:
        cdata[iteration, index, CONTACT_POSITION_X] /= norm_sum[index]
        cdata[iteration, index, CONTACT_POSITION_Y] /= norm_sum[index]
        cdata[iteration, index, CONTACT_POSITION_Z] /= norm_sum[index]


cdef inline void scale_forces(
    unsigned int iteration,
    unsigned int index,
    DTYPEv3 cdata,
    double imeters,
    double inewtons,
) nogil:
    """Scale forces"""
    cdata[iteration, index, CONTACT_REACTION_X] *= inewtons
    cdata[iteration, index, CONTACT_REACTION_Y] *= inewtons
    cdata[iteration, index, CONTACT_REACTION_Z] *= inewtons
    cdata[iteration, index, CONTACT_FRICTION_X] *= inewtons
    cdata[iteration, index, CONTACT_FRICTION_Y] *= inewtons
    cdata[iteration, index, CONTACT_FRICTION_Z] *= inewtons
    cdata[iteration, index, CONTACT_TOTAL_X] *= inewtons
    cdata[iteration, index, CONTACT_TOTAL_Y] *= inewtons
    cdata[iteration, index, CONTACT_TOTAL_Z] *= inewtons
    cdata[iteration, index, CONTACT_POSITION_X] *= imeters
    cdata[iteration, index, CONTACT_POSITION_Y] *= imeters
    cdata[iteration, index, CONTACT_POSITION_Z] *= imeters


cdef void postprocess_contacts(
    unsigned int iteration,
    DTYPEv3 cdata,
    unsigned int n_contact_sensors,
    DTYPEv1 norm_sum,
    double meters,
    double newtons,
) nogil:
    cdef unsigned int index
    cdef double imeters = 1./meters
    cdef double inewtons = 1./newtons
    for index in range(n_contact_sensors):
        normalize_forces_pos(
            iteration=iteration,
            index=index,
            cdata=cdata,
            norm_sum=norm_sum,
        )
        scale_forces(
            iteration=iteration,
            index=index,
            cdata=cdata,
            imeters=imeters,
            inewtons=inewtons,
        )


cpdef cycontacts2data(
    object physics,
    unsigned int iteration,
    ContactsArrayCy data,
    dict geompair2data,
    double meters,
    double newtons,
):
    """Contacts to data"""
    cdef unsigned int contact_i, index, n_contact_sensors=len(data.names)
    cdef unsigned int geom1, geom2
    cdef DTYPEv3 cdata = data.array
    cdef object model_ptr = physics.model.ptr
    cdef object data_ptr = physics.data.ptr
    cdef object contacts = physics.data.contact
    cdef DTYPEv1 norm_sum = np.zeros(data.array.shape[1], dtype=np.double)
    cdef np.ndarray[double, ndim=1] forcetorque = np.empty(6, dtype=np.double)
    for contact_i in range(len(contacts)):
        # Extract body index
        contact = contacts[contact_i]
        geom1 = contact.geom1
        geom2 = contact.geom2
        for pair, sign in [
                [(geom1, geom2), -1],
                [(geom2, geom1), +1],
                [(geom1, -1), -1],
                [(geom2, -1), +1],
        ]:
            if pair in geompair2data:
                cycontact2data(
                    iteration=iteration,
                    contact_i=contact_i,
                    index=geompair2data[pair],
                    model_ptr=model_ptr,
                    data_ptr=data_ptr,
                    contact=contact,
                    cdata=cdata,
                    forcetorque=forcetorque,
                    norm_sum=norm_sum,
                    sign=sign,
                )
    postprocess_contacts(
        iteration=iteration,
        cdata=cdata,
        n_contact_sensors=n_contact_sensors,
        norm_sum=norm_sum,
        meters=meters,
        newtons=newtons,
    )
