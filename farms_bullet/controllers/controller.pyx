"""Cython code"""

import time
# import numpy as np

cimport cython
cimport numpy as np

from libc.math cimport sin, cos, fabs
# from libc.stdlib cimport malloc, free
# from cython.parallel import prange


cpdef double[:] ode_oscillators_sparse(
    double time,
    CTYPE[:] state,
    AnimatData data
) nogil:
    """ODE"""
    cdef unsigned int i, i0, i1
    cdef unsigned int o_dim = data.network.oscillators.size[1]
    cdef double contact
    cdef double hydro_force
    cdef double[:] dstate = data.state.array[data.iteration+1][1]
    for i in range(o_dim):  # , nogil=True):
        # Intrinsic frequency
        dstate[i] = data.network.oscillators.array[0][i]
        # rate*(nominal_amplitude - amplitude)
        dstate[o_dim+i] = data.network.oscillators.array[1][i]*(
            data.network.oscillators.array[2][i] - state[o_dim+i]
        )
    for i in range(data.network.connectivity.size[0]):
        i0 = <unsigned int> (data.network.connectivity.array[i][0] + 0.5)
        i1 = <unsigned int> (data.network.connectivity.array[i][1] + 0.5)
        # amplitude_j*weight*sin(phase_j - phase_i - phase_bias)
        dstate[i0] += state[o_dim+i1]*data.network.connectivity.array[i][2]*sin(
            state[i1] - state[i0]
            - data.network.connectivity.array[i][3]
        )
    for i in range(data.network.contacts_connectivity.size[0]):
        i0 = <unsigned int> (
            data.network.contacts_connectivity.array[i][0] + 0.5
        )
        i1 = <unsigned int> (
            data.network.contacts_connectivity.array[i][1] + 0.5
        )
        # contact_weight*contact_force
        contact = (
            data.sensors.contacts.array[data.iteration][i1][0]**2
            + data.sensors.contacts.array[data.iteration][i1][1]**2
            + data.sensors.contacts.array[data.iteration][i1][2]**2
        )**0.5
        dstate[i0] += (
            data.network.contacts_connectivity.array[i][2]
            *(10*contact/(1+10*contact))
        )
    for i in range(data.network.hydro_connectivity.size[0]):
        i0 = <unsigned int> (
            data.network.hydro_connectivity.array[i][0] + 0.5
        )
        i1 = <unsigned int> (
            data.network.hydro_connectivity.array[i][1] + 0.5
        )
        # hydro_weight*hydro_force
        hydro_force = fabs(
            data.sensors.hydrodynamics.array[data.iteration][i1][1]
        )
        dstate[i0] += data.network.hydro_connectivity.array[i][2]*hydro_force
    for i in range(data.joints.size[1]):
        # rate*(joints_offset_desired - joints_offset)
        dstate[2*o_dim+i] = data.joints.array[1][i]*(
            data.joints.array[0][i] - state[2*o_dim+i]
        )
    return dstate


# cpdef void ode_oscillators_sparse_gradient(
#     CTYPE[:, :] jac,
#     CTYPE[:] state,
#     CTYPE[:, :] oscillators,
#     CTYPE[:, :] connectivity,
#     CTYPE[:, :] joints,
#     unsigned int o_dim,
#     unsigned int c_dim,
#     unsigned int j_dim
# ) nogil:
#     """ODE"""
#     cdef unsigned int i, i0, i1
#     for i in range(o_dim):  # , nogil=True):
#         # amplitude_i = rate_i*(nominal_amplitude_i - amplitude_i) gradient
#         jac[o_dim+i, o_dim+i] = -oscillators[1][i]
#     for i in range(c_dim):
#         i0 = <unsigned int> (connectivity[i][0] + 0.5)
#         i1 = <unsigned int> (connectivity[i][1] + 0.5)
#         # amplitude*weight*sin(phase_j - phase_i - phase_bias) gradient
#         jac[i0, i1] = connectivity[i][2]*sin(
#             state[i1] - state[i0] - connectivity[i][3]
#         ) + state[o_dim+i1]*connectivity[i][2]*cos(
#             state[i1] - state[i0] - connectivity[i][3]
#         )
#     for i in range(j_dim):
#         # rate*(joints_offset_desired - joints_offset) gradient
#         jac[2*o_dim+i, 2*o_dim+i] = -joints[1][i]
