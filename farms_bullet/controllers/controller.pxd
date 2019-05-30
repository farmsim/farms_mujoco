"""Cython controller code"""

cimport numpy as np

from ..animats.animat_data cimport (
    NetworkParameters
)

ctypedef double CTYPE
ctypedef np.float64_t DTYPE

cpdef void ode_oscillators_sparse(
    CTYPE[:] dstate,
    CTYPE[:] state,
    NetworkParameters params,
) nogil
