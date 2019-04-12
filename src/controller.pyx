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
from cython.parallel import prange


ctypedef double CTYPE
DTYPE = np.float64


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.profile(False)
@cython.nonecheck(False)
cpdef void odefun(CTYPE[:] dstate, CTYPE[:] state, CTYPE[:] freqs, CTYPE[:, :] weights, CTYPE[:, :] phi, unsigned int n_dim) nogil:
    """ODE"""
    cdef int i, j, n_dim_c = n_dim
    for i in prange(n_dim_c, nogil=True):
        dstate[i] = freqs[i]
        for j in range(n_dim_c):
            dstate[i] += weights[i, j]*sin(
                state[j] - state[i] + phi[i, j]
            )


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.profile(False)
@cython.nonecheck(False)
cpdef void rk4_ode(fun, float timestep, CTYPE[:] state, CTYPE[:] freqs, CTYPE[:, :] weights, CTYPE[:, :] phi, unsigned int n_dim):
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
    for i in prange(n_dim_c, nogil=True):
        k_1[i] = timestep*k_1[i]
        k_1_2[i] = state[i]+0.5*k_1[i]
    fun(k_2, k_1_2, freqs, weights, phi, n_dim_c)
    for i in prange(n_dim_c, nogil=True):
        k_2[i] = timestep*k_2[i]
        k_2_2[i] = state[i]+0.5*k_2[i]
    fun(k_3, k_2_2, freqs, weights, phi, n_dim_c)
    for i in prange(n_dim_c, nogil=True):
        k_3[i] = timestep*k_3[i]
        k_3_2[i] = state[i]+k_3[i]
    fun(k_4, k_3_2, freqs, weights, phi, n_dim_c)
    for i in prange(n_dim_c, nogil=True):
        k_4[i] = timestep*k_4[i]
        state[i] = state[i] + (k_1[i]+2*k_2[i]+2*k_3[i]+k_4[i])/6.
