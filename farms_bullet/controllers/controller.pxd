"""Cython controller code"""

cimport numpy as np

from ..animats.animat_data_cy cimport AnimatDataCy


ctypedef double CTYPE
ctypedef np.float64_t DTYPE


cpdef double[:] ode_oscillators_sparse(
    double time,
    CTYPE[:] state,
    AnimatDataCy data
) nogil
