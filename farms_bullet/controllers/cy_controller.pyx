"""Cython code"""

import time
# import numpy as np

cimport cython
cimport numpy as np

from libc.math cimport sin, cos
# from libc.stdlib cimport malloc, free
# from cython.parallel import prange


cpdef void ode_oscillators_sparse(
    CTYPE[:] dstate,
    CTYPE[:] state,
    NetworkParameters params
) nogil:
    """ODE"""
    cdef unsigned int i, i0, i1
    cdef unsigned int o_dim = params.oscillators.size[1]
    cdef double contact
    for i in range(o_dim):  # , nogil=True):
        # Intrinsic frequency
        dstate[i] = params.oscillators.array[0][i]
        # rate*(nominal_amplitude - amplitude)
        dstate[o_dim+i] = params.oscillators.array[1][i]*(
            params.oscillators.array[2][i] - state[o_dim+i]
        )
    for i in range(params.connectivity.size[0]):
        i0 = <unsigned int> (params.connectivity.array[i][0] + 0.5)
        i1 = <unsigned int> (params.connectivity.array[i][1] + 0.5)
        # amplitude*weight*sin(phase_j - phase_i - phase_bias)
        dstate[i0] += state[o_dim+i1]*params.connectivity.array[i][2]*sin(
            state[i1] - state[i0] - params.connectivity.array[i][3]
        )
    for i in range(params.contacts_connectivity.size[0]):
        i0 = <unsigned int> (params.contacts_connectivity.array[i][0] + 0.5)
        i1 = <unsigned int> (params.contacts_connectivity.array[i][1] + 0.5)
        # contact_weight*contact_force
        contact = (
            params.contacts.array[params.iteration][i1][0]**2
            + params.contacts.array[params.iteration][i1][1]**2
            + params.contacts.array[params.iteration][i1][2]**2
        )**0.5
        dstate[i0] += params.contacts_connectivity.array[i][2]*contact
    for i in range(params.joints.size[1]):
        # rate*(joints_offset_desired - joints_offset)
        dstate[2*o_dim+i] = params.joints.array[1][i]*(
            params.joints.array[0][i] - state[2*o_dim+i]
        )


cpdef void ode_oscillators_sparse_gradient(
    CTYPE[:, :] jac,
    CTYPE[:] state,
    CTYPE[:, :] oscillators,
    CTYPE[:, :] connectivity,
    CTYPE[:, :] joints,
    unsigned int o_dim,
    unsigned int c_dim,
    unsigned int j_dim
) nogil:
    """ODE"""
    cdef unsigned int i, i0, i1
    for i in range(o_dim):  # , nogil=True):
        # amplitude_i = rate_i*(nominal_amplitude_i - amplitude_i) gradient
        jac[o_dim+i, o_dim+i] = -oscillators[1][i]
    for i in range(c_dim):
        i0 = <unsigned int> (connectivity[i][0] + 0.5)
        i1 = <unsigned int> (connectivity[i][1] + 0.5)
        # amplitude*weight*sin(phase_j - phase_i - phase_bias) gradient
        jac[i0, i1] = connectivity[i][2]*sin(
            state[i1] - state[i0] - connectivity[i][3]
        ) + state[o_dim+i1]*connectivity[i][2]*cos(
            state[i1] - state[i0] - connectivity[i][3]
        )
    for i in range(j_dim):
        # rate*(joints_offset_desired - joints_offset) gradient
        jac[2*o_dim+i, 2*o_dim+i] = -joints[1][i]
