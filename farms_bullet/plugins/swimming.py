"""Swimming"""

import numpy as np
import pybullet

from ..simulations.simulation_options import SimulationUnitScaling


def viscous_swimming(
        iteration,
        data_gps,
        data_hydrodynamics,
        model,
        links,
        masses,
        **kwargs
):
    """Viscous swimming"""
    gravity = kwargs.pop("gravity", -9.81)
    units = kwargs.pop("units", SimulationUnitScaling())
    force_coefficients, torque_coefficients = kwargs.pop(
        "coefficients",
        [np.array([-1e-1, -1e0, -1e0]), np.array([-1e-2, -1e-2, -1e-2])]
    )
    for link_i, link in links:
        pos = data_gps.com_position(iteration, link_i)
        ori, lin_velocity, ang_velocity = (
            data_gps.urdf_orientation(iteration, link_i),
            data_gps.com_lin_velocity(iteration, link_i),
            data_gps.com_ang_velocity(iteration, link_i)
        )
        link_orientation_inv = np.linalg.inv(np.array(
            pybullet.getMatrixFromQuaternion(ori)
        ).reshape([3, 3]))
        link_velocity = np.dot(link_orientation_inv, lin_velocity)
        link_angular_velocity = np.dot(link_orientation_inv, ang_velocity)
        # Data
        buoyancy = 10*masses[link_i]*gravity*(0.1+pos[2])
        data_hydrodynamics[iteration, link_i, :3] = (
            force_coefficients*link_velocity
            + np.dot(link_orientation_inv, [0, 0, buoyancy])
        )
        data_hydrodynamics[iteration, link_i, 3:6] = (
            torque_coefficients*link_angular_velocity
        )
        # Forces
        pybullet.applyExternalForce(
            model,
            link,
            forceObj=(
                np.array(data_hydrodynamics[iteration, link_i, :3])
                *units.newtons
            ),
            posObj=[0, 0, 0],
            flags=pybullet.LINK_FRAME
        )
        pybullet.applyExternalTorque(
            model,
            link,
            torqueObj=(
                np.array(data_hydrodynamics[iteration, link_i, 3:6])
                *units.torques
            ),
            flags=pybullet.LINK_FRAME
        )
