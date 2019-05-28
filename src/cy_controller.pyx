# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=True
# cython: profile=True

"""Cython code"""


import time
import numpy as np

cimport cython
cimport numpy as np

from libc.math cimport sin, cos
# from libc.stdlib cimport malloc, free
# from cython.parallel import prange


ctypedef double CTYPE
DTYPE = np.float64


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
# @cython.profile(False)
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
) nogil:
    """ODE"""
    cdef unsigned int i, i0, i1
    cdef double contact
    for i in range(o_dim):  # , nogil=True):
        # Intrinsic frequency
        dstate[i] = oscillators[0][i]
        # rate*(nominal_amplitude - amplitude)
        dstate[o_dim+i] = oscillators[1][i]*(oscillators[2][i] - state[o_dim+i])
    for i in range(c_dim):
        i0 = <unsigned int> (connectivity[i][0] + 0.5)
        i1 = <unsigned int> (connectivity[i][1] + 0.5)
        # amplitude*weight*sin(phase_j - phase_i - phase_bias)
        dstate[i0] += state[o_dim+i1]*connectivity[i][2]*sin(
            state[i1] - state[i0] - connectivity[i][3]
        )
    for i in range(cc_dim):
        i0 = <unsigned int> (contacts_connectivity[i][0] + 0.5)
        i1 = <unsigned int> (contacts_connectivity[i][1] + 0.5)
        # amplitude*weight*sin(phase_j - phase_i - phase_bias)
        contact = (
            contacts[iteration][i1][0]**2
            + contacts[iteration][i1][1]**2
            + contacts[iteration][i1][2]**2
        )**0.5
        dstate[i0] += contacts_connectivity[i][2]*contact
    for i in range(j_dim):
        # rate*(joints_offset_desired - joints_offset)
        dstate[2*o_dim+i] = joints[1][i]*(joints[0][i] - state[2*o_dim+i])


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
# @cython.profile(False)
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


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
# @cython.profile(False)
cpdef void euler(
    fun,
    float timestep,
    CTYPE[:, :, :] state,
    unsigned int n_dim,
    unsigned int iteration,
    CTYPE[:, :] rk4_k,
    parameters
):
    """Runge-Kutta step integration"""
    cdef unsigned int i
    fun(state[iteration][1], state[iteration][0], *parameters)
    for i in range(n_dim):  # , nogil=True):
        state[iteration+1][0][i] = (
            state[iteration][0][i]
            + timestep*state[iteration][1][i]
        )


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
# @cython.profile(False)
cpdef void rk4(
    fun,
    float timestep,
    CTYPE[:, :, :] state,
    unsigned int n_dim,
    unsigned int iteration,
    CTYPE[:, :] rk4_k,
    parameters
):
    """Runge-Kutta step integration"""
    cdef unsigned int i
    fun(rk4_k[0], state[iteration][0], *parameters)
    for i in range(n_dim):  # , nogil=True):
        rk4_k[1][i] = state[iteration][0][i]+0.5*timestep*rk4_k[0][i]
    fun(rk4_k[2], rk4_k[1], *parameters)
    for i in range(n_dim):  # , nogil=True):
        rk4_k[3][i] = state[iteration][0][i]+0.5*timestep*rk4_k[2][i]
    fun(rk4_k[4], rk4_k[3], *parameters)
    for i in range(n_dim):  # , nogil=True):
        rk4_k[5][i] = state[iteration][0][i]+timestep*rk4_k[4][i]
    fun(rk4_k[6], rk4_k[5], *parameters)
    for i in range(n_dim):  # , nogil=True):
        state[iteration][1][i] = (
            (rk4_k[0][i] + 2*rk4_k[2][i] + 2*rk4_k[4][i] + rk4_k[6][i])/6.
        )
        state[iteration+1][0][i] = (
            state[iteration][0][i]
            + timestep*state[iteration][1][i]
        )
