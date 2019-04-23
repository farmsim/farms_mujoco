# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=True

"""Cython code"""


import time
import numpy as np

cimport cython
cimport numpy as np

from libc.math cimport sin
# from libc.stdlib cimport malloc, free
# from cython.parallel import prange


ctypedef double CTYPE
DTYPE = np.float64


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.profile(False)
@cython.nonecheck(False)
cpdef void odefun(
    CTYPE[:] dstate,
    CTYPE[:] state,
    CTYPE[:] freqs,
    CTYPE[:, :] weights,
    CTYPE[:, :] phi,
    unsigned int n_dim
) nogil:
    """ODE"""
    cdef int i, j, n_dim_c = n_dim
    for i in range(n_dim_c):  # , nogil=True):
        dstate[i] = freqs[i]
        for j in range(n_dim_c):
            dstate[i] += weights[i, j]*sin(
                state[j] - state[i] + phi[i, j]
            )


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.profile(False)
@cython.nonecheck(False)
cpdef void rk4_ode(
    fun,
    float timestep,
    CTYPE[:] state,
    CTYPE[:] freqs,
    CTYPE[:, :] weights,
    CTYPE[:, :] phi,
    unsigned int n_dim
):
    """Runge-Kutta step integration"""
    cdef int i, n_dim_c = n_dim
    cdef CTYPE[:] k_1 = np.empty([n_dim], dtype=DTYPE)
    cdef CTYPE[:] k_2 = np.empty([n_dim], dtype=DTYPE)
    cdef CTYPE[:] k_3 = np.empty([n_dim], dtype=DTYPE)
    cdef CTYPE[:] k_4 = np.empty([n_dim], dtype=DTYPE)
    cdef CTYPE[:] k_1_2 = np.empty([n_dim], dtype=DTYPE)
    cdef CTYPE[:] k_2_2 = np.empty([n_dim], dtype=DTYPE)
    cdef CTYPE[:] k_3_2 = np.empty([n_dim], dtype=DTYPE)
    fun(k_1, state, freqs, weights, phi, n_dim_c)
    for i in range(n_dim_c):  # , nogil=True):
        k_1[i] = timestep*k_1[i]
        k_1_2[i] = state[i]+0.5*k_1[i]
    fun(k_2, k_1_2, freqs, weights, phi, n_dim_c)
    for i in range(n_dim_c):  # , nogil=True):
        k_2[i] = timestep*k_2[i]
        k_2_2[i] = state[i]+0.5*k_2[i]
    fun(k_3, k_2_2, freqs, weights, phi, n_dim_c)
    for i in range(n_dim_c):  # , nogil=True):
        k_3[i] = timestep*k_3[i]
        k_3_2[i] = state[i]+k_3[i]
    fun(k_4, k_3_2, freqs, weights, phi, n_dim_c)
    for i in range(n_dim_c):  # , nogil=True):
        k_4[i] = timestep*k_4[i]
        state[i] = state[i] + (k_1[i]+2*k_2[i]+2*k_3[i]+k_4[i])/6.


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.profile(False)
@cython.nonecheck(False)
cpdef void odefun_sparse(
    CTYPE[:] dstate,
    CTYPE[:] state,
    CTYPE[:] freqs,
    unsigned int[:, :] connectivity,
    CTYPE[:, :] connection,
    unsigned int n_dim,
    unsigned int c_dim
) nogil:
    """ODE"""
    cdef int i, j, n_dim_c = n_dim, c_dim_c = c_dim
    for i in range(n_dim_c):  # , nogil=True):
        dstate[i] = freqs[i]
    for j in range(c_dim_c):
        dstate[connectivity[j][0]] += connection[j][0]*sin(
            state[connectivity[j][1]]
            - state[connectivity[j][0]]
            - connection[j][1]
        )


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.profile(False)
@cython.nonecheck(False)
cpdef void rk4_ode_sparse(
    fun,
    float timestep,
    CTYPE[:] state,
    CTYPE[:] freqs,
    unsigned int[:, :] connectivity,
    CTYPE[:, :] connection,
    unsigned int n_dim,
    unsigned int c_dim
):
    """Runge-Kutta step integration"""
    cdef int i, n_dim_c = n_dim, c_dim_c = c_dim
    cdef CTYPE[:] k_1 = np.empty([n_dim], dtype=DTYPE)
    cdef CTYPE[:] k_2 = np.empty([n_dim], dtype=DTYPE)
    cdef CTYPE[:] k_3 = np.empty([n_dim], dtype=DTYPE)
    cdef CTYPE[:] k_4 = np.empty([n_dim], dtype=DTYPE)
    cdef CTYPE[:] k_1_2 = np.empty([n_dim], dtype=DTYPE)
    cdef CTYPE[:] k_2_2 = np.empty([n_dim], dtype=DTYPE)
    cdef CTYPE[:] k_3_2 = np.empty([n_dim], dtype=DTYPE)
    fun(k_1, state, freqs, connectivity, connection, n_dim_c, c_dim_c)
    for i in range(n_dim_c):  # , nogil=True):
        k_1[i] = timestep*k_1[i]
        k_1_2[i] = state[i]+0.5*k_1[i]
    fun(k_2, k_1_2, freqs, connectivity, connection, n_dim_c, c_dim_c)
    for i in range(n_dim_c):  # , nogil=True):
        k_2[i] = timestep*k_2[i]
        k_2_2[i] = state[i]+0.5*k_2[i]
    fun(k_3, k_2_2, freqs, connectivity, connection, n_dim_c, c_dim_c)
    for i in range(n_dim_c):  # , nogil=True):
        k_3[i] = timestep*k_3[i]
        k_3_2[i] = state[i]+k_3[i]
    fun(k_4, k_3_2, freqs, connectivity, connection, n_dim_c, c_dim_c)
    for i in range(n_dim_c):  # , nogil=True):
        k_4[i] = timestep*k_4[i]
        state[i] = state[i] + (k_1[i]+2*k_2[i]+2*k_3[i]+k_4[i])/6.


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.profile(False)
@cython.nonecheck(False)
cpdef void ode_amplitude(
    CTYPE[:] dstate,
    CTYPE[:] state,
    unsigned int n_dim,
    CTYPE[:] rate,
    CTYPE[:] desired
) nogil:
    """ODE"""
    cdef int i, n_dim_c = n_dim
    for i in range(n_dim_c):  # , nogil=True):
        dstate[i] = rate[i]*(desired[i] - state[i])


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.profile(False)
@cython.nonecheck(False)
cpdef void ode_oscillators_sparse(
    CTYPE[:] dstate,
    CTYPE[:] state,
    CTYPE[:, :] oscillators,
    CTYPE[:, :] connectivity,
    CTYPE[:, :] joints,
    unsigned int o_dim,
    unsigned int c_dim
):
    """ODE"""
    cdef unsigned int i, j, i0, i1
    for i in range(o_dim):  # , nogil=True):
        dstate[i] = oscillators[0][i]
        dstate[i+o_dim] = (
            oscillators[1][i]*(oscillators[2][i] - state[i+o_dim])
        )
    for j in range(c_dim):
        i0 = <unsigned int> (connectivity[j][0] + 0.5)
        i1 = <unsigned int> (connectivity[j][1] + 0.5)
        dstate[i0] += connectivity[j][2]*sin(
            state[i1] - state[i0] - connectivity[j][3]
        )


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.profile(False)
@cython.nonecheck(False)
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


# @cython.boundscheck(False)  # Deactivate bounds checking
# @cython.wraparound(False)   # Deactivate negative indexing.
# @cython.profile(False)
# @cython.nonecheck(False)
# cdef class ODESolver(object):
#     """ODE solver"""
#     cdef unsigned int n_dim
#     cdef double timestep
#     cdef CTYPE[:] k_1
#     cdef CTYPE[:] k_2
#     cdef CTYPE[:] k_3
#     cdef CTYPE[:] k_4
#     cdef CTYPE[:] k_1_2
#     cdef CTYPE[:] k_2_2
#     cdef CTYPE[:] k_3_2
#     cdef CTYPE[:, :] rk4_k

#     def __cinit__(self, double timestep, unsigned int n_dim):
#         self.timestep = timestep
#         self.n_dim = n_dim
#         k_1 = np.zeros([self.n_dim], dtype=DTYPE)
#         k_2 = np.zeros([self.n_dim], dtype=DTYPE)
#         k_3 = np.zeros([self.n_dim], dtype=DTYPE)
#         k_4 = np.zeros([self.n_dim], dtype=DTYPE)
#         k_1_2 = np.zeros([self.n_dim], dtype=DTYPE)
#         k_2_2 = np.zeros([self.n_dim], dtype=DTYPE)
#         k_3_2 = np.zeros([self.n_dim], dtype=DTYPE)

#     @cython.boundscheck(False)  # Deactivate bounds checking
#     @cython.wraparound(False)   # Deactivate negative indexing.
#     @cython.profile(False)
#     @cython.nonecheck(False)
#     cdef void euler(
#         self,
#         fun,
#         CTYPE[:] state,
#         CTYPE[:] dstate,
#         parameters
#     ):
#         """Euler step integration"""
#         cdef unsigned int i
#         fun(dstate, state, self.n_dim, *parameters)
#         for i in range(self.n_dim):  # , nogil=True):
#             state[i] = state[i] + self.timestep*dstate[i]

#     @cython.boundscheck(False)  # Deactivate bounds checking
#     @cython.wraparound(False)   # Deactivate negative indexing.
#     @cython.profile(False)
#     @cython.nonecheck(False)
#     cdef void rk4(
#         self,
#         fun,
#         CTYPE[:] state,
#         CTYPE[:] dstate,
#         parameters
#     ):
#         """Runge-Kutta step integration"""
#         cdef int i
#         fun(self.k_1, state, self.n_dim, *parameters)
#         for i in range(self.n_dim):  # , nogil=True):
#             self.k_1_2[i] = state[i]+0.5*self.timestep*self.k_1[i]
#         fun(self.k_2, self.k_1_2, self.n_dim, *parameters)
#         for i in range(self.n_dim):  # , nogil=True):
#             self.k_2_2[i] = state[i]+0.5*self.timestep*self.k_2[i]
#         fun(self.k_3, self.k_2_2, self.n_dim, *parameters)
#         for i in range(self.n_dim):  # , nogil=True):
#             self.k_3_2[i] = state[i]+self.timestep*self.k_3[i]
#         fun(self.k_4, self.k_3_2, self.n_dim, *parameters)
#         for i in range(self.n_dim):  # , nogil=True):
#             dstate[i] = (self.k_1[i]+2*self.k_2[i]+2*self.k_3[i]+self.k_4[i])/6.
#             state[i] = state[i] + self.timestep*dstate[i]

#     @cython.boundscheck(False)  # Deactivate bounds checking
#     @cython.wraparound(False)   # Deactivate negative indexing.
#     @cython.profile(False)
#     @cython.nonecheck(False)
#     cdef void rk4_2(
#         self,
#         fun,
#         CTYPE[:] state,
#         CTYPE[:] dstate,
#         parameters
#     ):
#         """Runge-Kutta step integration"""
#         cdef int i
#         fun(self.rk4_k[0], state, self.n_dim, *parameters)
#         for i in range(self.n_dim):  # , nogil=True):
#             self.rk4_k[1][i] = state[i]+0.5*self.timestep*self.rk4_k[0][i]
#         fun(self.rk4_k[2], self.rk4_k[1], self.n_dim, *parameters)
#         for i in range(self.n_dim):  # , nogil=True):
#             self.rk4_k[3][i] = state[i]+0.5*self.timestep*self.rk4_k[2][i]
#         fun(self.rk4_k[4], self.rk4_k[3], self.n_dim, *parameters)
#         for i in range(self.n_dim):  # , nogil=True):
#             self.rk4_k[5][i] = state[i]+self.timestep*self.rk4_k[4][i]
#         fun(self.rk4_k[6], self.rk4_k[5], self.n_dim, *parameters)
#         for i in range(self.n_dim):  # , nogil=True):
#             dstate[i] = (self.rk4_k[0][i]+2*self.rk4_k[2][i]+2*self.rk4_k[3][i]+self.rk4_k[5][i])/6.
#             state[i] = state[i] + self.timestep*dstate[i]


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.profile(False)
@cython.nonecheck(False)
cpdef inline void ode_oscillators(
    CTYPE[:] dstate,
    CTYPE[:] state,
    unsigned int n_dim,
    CTYPE[:] freqs,
    unsigned int[:, :] connectivity,
    CTYPE[:, :] connection,
    unsigned int c_dim,
    CTYPE[:] rate,
    CTYPE[:] amplitude_desired
) nogil:
    """Oscillator ODE"""
    cdef unsigned int i
    for i in range(n_dim):  # , nogil=True):
        dstate[i] = freqs[i]
        dstate[i+n_dim] = rate[i]*(amplitude_desired[i] - state[i+n_dim])
    for i in range(c_dim):
        dstate[connectivity[i][0]] += state[i+n_dim]*connection[i][0]*sin(
            state[connectivity[i][1]]
            - state[connectivity[i][0]]
            - connection[i][1]
        )


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.profile(False)
@cython.nonecheck(False)
cdef inline void rk4_2(
    fun,
    double timestep,
    CTYPE[:] state,
    CTYPE[:] dstate,
    unsigned int n_dim,
    CTYPE [:, :] rk4_k,
    parameters
):
    """Runge-Kutta step integration"""
    cdef int i
    fun(rk4_k[0], state, n_dim, *parameters)
    for i in range(n_dim):  # , nogil=True):
        rk4_k[1][i] = state[i]+0.5*timestep*rk4_k[0][i]
    fun(rk4_k[2], rk4_k[1], n_dim, *parameters)
    for i in range(n_dim):  # , nogil=True):
        rk4_k[3][i] = state[i]+0.5*timestep*rk4_k[2][i]
    fun(rk4_k[4], rk4_k[3], n_dim, *parameters)
    for i in range(n_dim):  # , nogil=True):
        rk4_k[5][i] = state[i]+timestep*rk4_k[4][i]
    fun(rk4_k[6], rk4_k[5], n_dim, *parameters)
    for i in range(n_dim):  # , nogil=True):
        dstate[i] = (rk4_k[0][i]+2*rk4_k[2][i]+2*rk4_k[4][i]+rk4_k[6][i])/6.
        state[i] = state[i] + timestep*dstate[i]


cdef class OscillatorNetwork:
    """Network"""
    cdef public double timestep
    cdef public CTYPE[:, :] state
    cdef public CTYPE[:, :] dstate
    cdef public CTYPE [:, :] rk4_k
    cdef public unsigned int n_dim
    cdef public unsigned int c_dim
    cdef public CTYPE[:] freqs
    cdef public unsigned int[:, :] connectivity
    cdef public CTYPE[:, :] connection
    cdef public CTYPE[:] rate
    cdef public CTYPE[:] amplitude_desired

    def __cinit__(
            self,
            double timestep,
            unsigned int n_iterations,
            unsigned int n_dim
    ):
        self.n_dim = n_dim
        self.timestep = timestep
        self.state = np.empty([n_iterations, n_dim], dtype=DTYPE)
        self.dstate = np.empty([n_iterations, n_dim], dtype=DTYPE)
        self.ode = ode_oscillators
        self.params.n_dim = n_dim
        self.params.n_dim = np.empty([n_iterations, n_dim], dtype=DTYPE)

    cpdef step_rk(self, unsigned int iteration):
        """ODE integration step"""
        rk4_2(
            self.ode,
            self.timestep,
            self.state[iteration, :],
            self.dstate[iteration, :],
            self.n_dim,
            self.rk4_k,
            [
                self.freqs,
                self.connectivity,
                self.connection,
                self.c_dim,
                self.rate,
                self.amplitude_desired
            ]
        )
