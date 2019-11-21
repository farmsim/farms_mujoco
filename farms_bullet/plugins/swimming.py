"""Swimming"""

import numpy as np
import pybullet

from ..simulations.simulation_options import SimulationUnitScaling


def viscous_forces(
        iteration,
        data_gps,
        data_hydrodynamics,
        links,
        masses,
        **kwargs
):
    """Viscous swimming"""
    gravity = kwargs.pop("gravity", -9.81)
    force_coefficients, torque_coefficients = kwargs.pop(
        "coefficients",
        [np.array([-1e-1, -1e0, -1e0]), np.array([-1e-2, -1e-2, -1e-2])]
    )
    use_buoyancy = kwargs.pop("buoyancy", True)
    buoyancy = np.zeros(3)
    for link_i in links:
        ori, lin_velocity, ang_velocity = (
            data_gps.urdf_orientation(iteration, link_i),
            data_gps.com_lin_velocity(iteration, link_i),
            data_gps.com_ang_velocity(iteration, link_i)
        )
        if not any(ori):
            continue
        # Compute velocity in local frame
        link_orientation_inv = np.array(
            pybullet.getMatrixFromQuaternion(ori)
        ).reshape([3, 3]).T
        link_velocity = np.dot(link_orientation_inv, lin_velocity)
        link_angular_velocity = np.dot(link_orientation_inv, ang_velocity)
        # Data
        if use_buoyancy:
            buoyancy = np.dot(
                link_orientation_inv,
                [0, 0, 10*masses[link_i]*gravity*(
                    0.1+data_gps.com_position(iteration, link_i)[2]
                )]
            )
        data_hydrodynamics[iteration, link_i, :3] = (
            np.sign(link_velocity)*force_coefficients*link_velocity**2
            + buoyancy
        )
        data_hydrodynamics[iteration, link_i, 3:6] = (
            np.sign(link_angular_velocity)*torque_coefficients*link_angular_velocity**2
        )


def swimming_motion(
        iteration,
        data_hydrodynamics,
        model,
        links,
        link_frame=True,
        **kwargs
):
    """Swimming motion"""
    units = kwargs.pop("units", SimulationUnitScaling())
    for link_i, link in links:
        pybullet.applyExternalForce(
            model,
            link,
            forceObj=(
                np.array(data_hydrodynamics[iteration, link_i, :3])
                *units.newtons
            ),
            posObj=[0, 0, 0],
            flags=pybullet.LINK_FRAME if link_frame else pybullet.WORLD_FRAME
        )
        pybullet.applyExternalTorque(
            model,
            link,
            torqueObj=(
                np.array(data_hydrodynamics[iteration, link_i, 3:6])
                *units.torques
            ),
            flags=pybullet.LINK_FRAME if link_frame else pybullet.WORLD_FRAME
        )


def swimming_debug(iteration, data_gps, links):
    """Swimming debug"""
    for link_i, _ in links:
        joint = np.array(data_gps.urdf_position(iteration, link_i))
        if link_i == 11:
            print("RBP position: {}".format(np.array(joint)))
        joint_ori = np.array(data_gps.urdf_orientation(iteration, link_i))
        # com_ori = np.array(data_gps.com_orientation(iteration, link_i))
        ori_joint = np.array(
            pybullet.getMatrixFromQuaternion(joint_ori)
        ).reshape([3, 3])
        # ori_com = np.array(
        #     pybullet.getMatrixFromQuaternion(com_ori)
        # ).reshape([3, 3])
        # ori = np.dot(ori_joint, ori_com)
        axis = 0.05
        offset_x = np.dot(ori_joint, np.array([axis, 0, 0]))
        offset_y = np.dot(ori_joint, np.array([0, axis, 0]))
        offset_z = np.dot(ori_joint, np.array([0, 0, axis]))
        print("SPH position: {}".format(np.array(joint)))
        for i, offset in enumerate([offset_x, offset_y, offset_z]):
            color = np.zeros(3)
            color[i] = 1
            pybullet.addUserDebugLine(
                joint,
                joint + offset,
                lineColorRGB=color,
                lineWidth=5,
                lifeTime=1,
            )
