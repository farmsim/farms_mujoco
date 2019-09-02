"""Cython controller code"""

cimport numpy as np

from ..animats.animat_data_cy cimport AnimatData


ctypedef double CTYPE
ctypedef np.float64_t DTYPE


cpdef double[:] ode_oscillators_sparse(
    double time,
    CTYPE[:] state,
    AnimatData data
) nogil
