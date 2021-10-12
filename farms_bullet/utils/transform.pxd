"""Transform"""

include 'types.pxd'


cdef void quat_conj(
    DTYPEv1 quat,
    DTYPEv1 out,
) nogil

cdef void quat_mult(
    DTYPEv1 q0,
    DTYPEv1 q1,
    DTYPEv1 out,
    bint full=*,
) nogil

cdef void quat_rot(
    DTYPEv1 vector,
    DTYPEv1 quat,
    DTYPEv1 quat_c,
    DTYPEv1 tmp4,
    DTYPEv1 out,
) nogil
