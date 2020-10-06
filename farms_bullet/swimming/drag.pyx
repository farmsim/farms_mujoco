"""Drag forces"""

from farms_data.sensors.data_cy cimport HydrodynamicsArrayCy, GpsArrayCy

import pybullet
import numpy as np


cpdef link_swimming_info(GpsArrayCy data_gps, iteration, sensor_i):
    """Link swimming information

    Times:
    - 10.369 [s]
    - 9.403 [s]
    - 8.972 [s]
    - 7.815 [s]
    - 7.204 [s]
    - 4.507 [s]
    - 4.304 [s]
    """

    # Declarations
    zeros = [0, 0, 0]
    quat_unit = [0, 0, 0, 1]

    # Orientations
    ori_urdf = np.array(
        data_gps.urdf_orientation_cy(iteration, sensor_i),
        copy=False,
    ).tolist()
    ori_com = np.array(
        data_gps.com_orientation_cy(iteration, sensor_i),
        copy=False,
    ).tolist()
    global2com = pybullet.invertTransform(zeros, ori_com)
    # urdf2global = (zeros, ori_urdf)

    # Velocities in global frame
    lin_velocity = np.array(
        data_gps.com_lin_velocity_cy(iteration, sensor_i),
        copy=False,
    ).tolist()
    ang_velocity = np.array(
        data_gps.com_ang_velocity_cy(iteration, sensor_i),
        copy=False,
    ).tolist()

    # Compute velocity in CoM frame
    link_velocity = np.array(pybullet.multiplyTransforms(
        *global2com,
        lin_velocity,
        quat_unit,
    )[0])
    link_angular_velocity = np.array(pybullet.multiplyTransforms(
        *global2com,
        ang_velocity,
        quat_unit,
    )[0])
    urdf2com = pybullet.multiplyTransforms(
        *global2com,
        zeros,  # *urdf2global,
        ori_urdf,  # *urdf2global,
    )
    return (
        link_velocity,
        link_angular_velocity,
        global2com,
        urdf2com,
    )


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
