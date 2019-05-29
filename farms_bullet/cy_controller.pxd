# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=True
# cython: profile=True

"""Cython controller code"""

cimport numpy as np

ctypedef double CTYPE
ctypedef np.float64_t DTYPE

cpdef void ode_oscillators_sparse(
    CTYPE[:] dstate,
    CTYPE[:] state,
    CTYPE[:, :] oscillators,
    CTYPE[:, :] connectivity,
    CTYPE[:, :] joints,
    CTYPE[:, :, :] contacts,
    CTYPE[:, :] contacts_connectivity,
    unsigned int o_dim,
    unsigned int c_dim,
    unsigned int j_dim,
    unsigned int contacts_dim,
    unsigned int cc_dim,
    unsigned int iteration
) nogil
