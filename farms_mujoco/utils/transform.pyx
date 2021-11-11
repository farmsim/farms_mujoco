"""Transform"""


cdef void quat_conj(
    DTYPEv1 quat,
    DTYPEv1 out,
) nogil:
    """Quaternion conjugate"""
    cdef unsigned int i
    for i in range(3):
        out[i] = -quat[i]  # x, y, z
    out[3] = quat[3]  # w


cdef void quat_mult(
    DTYPEv1 q0,
    DTYPEv1 q1,
    DTYPEv1 out,
    bint full=1,
) nogil:
    """Hamilton product of two quaternions out = q0*q1"""
    out[0] = q0[3]*q1[0] + q0[0]*q1[3] + q0[1]*q1[2] - q0[2]*q1[1]  # x
    out[1] = q0[3]*q1[1] - q0[0]*q1[2] + q0[1]*q1[3] + q0[2]*q1[0]  # y
    out[2] = q0[3]*q1[2] + q0[0]*q1[1] - q0[1]*q1[0] + q0[2]*q1[3]  # z
    if full:
        out[3] = q0[3]*q1[3] - q0[0]*q1[0] - q0[1]*q1[1] - q0[2]*q1[2]  # w


cdef void quat_rot(
    DTYPEv1 vector,
    DTYPEv1 quat,
    DTYPEv1 quat_c,
    DTYPEv1 tmp4,
    DTYPEv1 out,
) nogil:
    """Quaternion rotation

    :param vector: Vector to rotate
    :param quat: Quaternion rotation
    :param quat_c: Returned quaternion conjugate
    :param tmp4: Temporary quaternion
    :param out: Rotated vector

    """
    quat_c[3] = 0
    quat_c[0] = vector[0]
    quat_c[1] = vector[1]
    quat_c[2] = vector[2]
    quat_mult(quat, quat_c, tmp4)
    quat_conj(quat, quat_c)
    quat_mult(tmp4, quat_c, out, full=0)
