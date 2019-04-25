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
cdef class ODESolver(object):
    """ODE solver"""
    cdef unsigned int n_dim
    cdef double timestep
    cdef CTYPE[:] k_1
    cdef CTYPE[:] k_2
    cdef CTYPE[:] k_3
    cdef CTYPE[:] k_4
    cdef CTYPE[:] k_1_2
    cdef CTYPE[:] k_2_2
    cdef CTYPE[:] k_3_2
    cdef CTYPE[:, :] rk4_k

    def __cinit__(self, double timestep, unsigned int n_dim):
        self.timestep = timestep
        self.n_dim = n_dim
        k_1 = np.zeros([self.n_dim], dtype=DTYPE)
        k_2 = np.zeros([self.n_dim], dtype=DTYPE)
        k_3 = np.zeros([self.n_dim], dtype=DTYPE)
        k_4 = np.zeros([self.n_dim], dtype=DTYPE)
        k_1_2 = np.zeros([self.n_dim], dtype=DTYPE)
        k_2_2 = np.zeros([self.n_dim], dtype=DTYPE)
        k_3_2 = np.zeros([self.n_dim], dtype=DTYPE)

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.profile(False)
    @cython.nonecheck(False)
    cdef void euler(
        self,
        fun,
        CTYPE[:] state,
        CTYPE[:] dstate,
        parameters
    ):
        """Euler step integration"""
        cdef unsigned int i
        fun(dstate, state, self.n_dim, *parameters)
        for i in range(self.n_dim):  # , nogil=True):
            state[i] = state[i] + self.timestep*dstate[i]

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.profile(False)
    @cython.nonecheck(False)
    cdef void rk4(
        self,
        fun,
        CTYPE[:] state,
        CTYPE[:] dstate,
        parameters
    ):
        """Runge-Kutta step integration"""
        cdef int i
        fun(self.k_1, state, self.n_dim, *parameters)
        for i in range(self.n_dim):  # , nogil=True):
            self.k_1_2[i] = state[i]+0.5*self.timestep*self.k_1[i]
        fun(self.k_2, self.k_1_2, self.n_dim, *parameters)
        for i in range(self.n_dim):  # , nogil=True):
            self.k_2_2[i] = state[i]+0.5*self.timestep*self.k_2[i]
        fun(self.k_3, self.k_2_2, self.n_dim, *parameters)
        for i in range(self.n_dim):  # , nogil=True):
            self.k_3_2[i] = state[i]+self.timestep*self.k_3[i]
        fun(self.k_4, self.k_3_2, self.n_dim, *parameters)
        for i in range(self.n_dim):  # , nogil=True):
            dstate[i] = (self.k_1[i]+2*self.k_2[i]+2*self.k_3[i]+self.k_4[i])/6.
            state[i] = state[i] + self.timestep*dstate[i]

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.profile(False)
    @cython.nonecheck(False)
    cdef void rk4_2(
        self,
        fun,
        CTYPE[:] state,
        CTYPE[:] dstate,
        parameters
    ):
        """Runge-Kutta step integration"""
        cdef int i
        fun(self.rk4_k[0], state, self.n_dim, *parameters)
        for i in range(self.n_dim):  # , nogil=True):
            self.rk4_k[1][i] = state[i]+0.5*self.timestep*self.rk4_k[0][i]
        fun(self.rk4_k[2], self.rk4_k[1], self.n_dim, *parameters)
        for i in range(self.n_dim):  # , nogil=True):
            self.rk4_k[3][i] = state[i]+0.5*self.timestep*self.rk4_k[2][i]
        fun(self.rk4_k[4], self.rk4_k[3], self.n_dim, *parameters)
        for i in range(self.n_dim):  # , nogil=True):
            self.rk4_k[5][i] = state[i]+self.timestep*self.rk4_k[4][i]
        fun(self.rk4_k[6], self.rk4_k[5], self.n_dim, *parameters)
        for i in range(self.n_dim):  # , nogil=True):
            dstate[i] = (self.rk4_k[0][i]+2*self.rk4_k[2][i]+2*self.rk4_k[3][i]+self.rk4_k[5][i])/6.
            state[i] = state[i] + self.timestep*dstate[i]


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
