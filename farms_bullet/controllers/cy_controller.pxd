# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=True
# cython: profile=True

"""Cython controller code"""

cimport numpy as np

from .cy_animat_data cimport (
    NetworkParameters
)

ctypedef double CTYPE
ctypedef np.float64_t DTYPE

cpdef void ode_oscillators_sparse(
    CTYPE[:] dstate,
    CTYPE[:] state,
    NetworkParameters params,
) nogil
