"""Drag forces"""

from farms_data.sensors.data_cy cimport HydrodynamicsArrayCy

import pybullet
import numpy as np


cpdef swimming_motion(
        unsigned int iteration,
        HydrodynamicsArrayCy data_hydrodynamics,
        int model,
        list links,
        dict links_map,
        link_frame,
        units,
        pos=np.zeros(3)
):
    """Swimming motion"""
    cdef int link_id
    cdef str link_name
    cdef unsigned int i, sensor_i, flags
    cdef double[:] hydro
    cdef double hydro_force[3], hydro_torque[3]
    cdef double newtons, torques
    newtons = units.newtons
    torques = units.torques
    flags = pybullet.LINK_FRAME if link_frame else pybullet.WORLD_FRAME
    for link in links:
        # pybullet.LINK_FRAME applies force in inertial frame, not URDF frame
        sensor_i = data_hydrodynamics.names.index(link.name)
        link_id = links_map[link.name]
        hydro = data_hydrodynamics.array[iteration, sensor_i]
        for i in range(3):
            hydro_force[i] = hydro[i]*newtons
            hydro_torque[i] = hydro[i+3]*torques
        pybullet.applyExternalForce(
            model,
            link_id,
            forceObj=np.array(hydro_force),
            posObj=pos,  # pybullet.getDynamicsInfo(model, link)[3]
            flags=flags,
        )
        pybullet.applyExternalTorque(
            model,
            link_id,
            torqueObj=np.array(hydro_torque),
            flags=flags,
        )
