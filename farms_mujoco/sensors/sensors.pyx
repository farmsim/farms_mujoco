"""Cython sensors"""

import numpy as np
cimport numpy as np

try:
    from farms_muscle import rigid_tendon as rt
except:
    from libc import printf
    printf("farms_muscle not installed")
from mujoco import mj_contactForce

from libc.math cimport sqrt, abs


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
    cdef unsigned int i
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
    mj_contactForce(model_ptr, data_ptr, contact_i, forcetorque)
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
    cdef int sign
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
                index = geompair2data[pair]
                cycontact2data(
                    iteration=iteration,
                    contact_i=contact_i,
                    index=index,
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


cpdef cymusclesensors2data(
    object physics,
    unsigned int iteration,
    MusclesArrayCy data,
    np.ndarray musclesensor2data,
    double meters,
    double velocity,
    double newtons,
):
    """ Compute and update muscle states, spindle and golgi tendon
    feedbacks """
    cdef DTYPEv3 cdata = data.array
    cdef object model_ptr = physics.model.ptr
    cdef object data_ptr = physics.data.ptr
    cdef unsigned int n_muscles = len(data.names)
    cdef unsigned int mindex
    cdef int[6] objids
    for mindex in range(n_muscles):
        objids = musclesensor2data[mindex]
        cymusclesensor2data(
            iteration=iteration,
            index=mindex,
            objids=objids,
            model_ptr=model_ptr,
            data_ptr=data_ptr,
            cdata=cdata,
            imeters=1/meters,
            ivelocity=1/velocity,
            inewtons=1/newtons,
        )


cdef void cymusclesensor2data(
    unsigned int iteration,
    unsigned int index,
    int [6] objids,
    object model_ptr,
    object data_ptr,
    DTYPEv3 cdata,
    double imeters,
    double ivelocity,
    double inewtons,
):
    # Declarations
    # type Ia feedback constants
    cdef double Ia_kv, Ia_pv, Ia_k_dI, Ia_k_nI, Ia_const_I,
    # type II feedback constants
    cdef double II_k_dII, II_k_nII, II_const_II
    # type Ib feedback constants
    cdef double Ib_kF
    # l_opt and v_max
    cdef double l_opt, l_slack, v_max, f_max, alpha_opt
    # muscle states
    cdef double alpha, l_ce, v_ce
    cdef double act = data_ptr.act[objids[0]]
    cdef double l_mtu = data_ptr.actuator_length[objids[1]]*imeters
    cdef double v_mtu = data_ptr.actuator_velocity[objids[2]]*ivelocity
    cdef double force = data_ptr.actuator_force[objids[3]]*inewtons
    # muscle params
    cdef double[:] gainprm = model_ptr.actuator_gainprm[objids[4]]
    f_max = gainprm[0]*inewtons
    l_opt = gainprm[1]*imeters
    l_slack = gainprm[2]*imeters
    v_max = gainprm[3]*ivelocity
    alpha_opt = gainprm[4]
    # act, alpha, l_ce, v_ce
    alpha = rt.c_pennation_angle(l_mtu, l_opt, l_slack, alpha_opt)
    l_ce = rt.c_fiber_length(l_mtu, l_slack, alpha)/l_opt
    v_ce = rt.c_fiber_velocity(v_mtu, alpha)/v_max
    cdata[iteration, index, MUSCLE_ACTIVATION] = act
    cdata[iteration, index, MUSCLE_PENNATION_ANGLE] = alpha
    cdata[iteration, index, MUSCLE_FIBER_LENGTH] = l_ce
    cdata[iteration, index, MUSCLE_FIBER_VELOCITY] = v_ce
    # forces
    cdata[iteration, index, MUSCLE_ACTIVE_FORCE] = rt.c_active_force(
        l_ce, v_ce, alpha
    )
    cdata[iteration, index, MUSCLE_PASSIVE_FORCE] = rt.c_passive_force(
        l_ce, v_ce, alpha
    )
    # muscle spindles and golgi tendon feedbacks
    # IA
    Ia_kv , Ia_pv , Ia_k_dI , Ia_k_nI , Ia_const_I = (
        model_ptr.actuator_user[objids[5]][:5]
    )
    cdata[iteration, index, MUSCLE_IA_FEEDBACK] = (
        Ia_kv*abs(v_ce)**Ia_pv + Ia_k_dI*l_ce + Ia_k_nI*act + Ia_const_I
    )
    # II
    II_k_dII, II_k_nII, II_const_II = (
        model_ptr.actuator_user[objids[5]][5:8]
    )
    cdata[iteration, index, MUSCLE_II_FEEDBACK] = (
        II_k_dII*l_ce + II_k_nII*act + II_const_II
    )
    # IB
    Ib_kF = model_ptr.actuator_user[objids[5]][8]
    cdata[iteration, index, MUSCLE_IB_FEEDBACK] = Ib_kF*-force/f_max
