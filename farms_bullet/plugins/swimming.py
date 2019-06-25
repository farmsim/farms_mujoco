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
        **kwargs
):
    """Viscous swimming"""
    units = kwargs.pop("units", SimulationUnitScaling())
    force_coefficients, torque_coefficients = kwargs.pop(
        "coefficients",
        [np.array([-1e-1, -1e0, -1e0]), np.array([-1e-2, -1e-2, -1e-2])]
    )
    for link_i, link in links:
        ori, lin_velocity, ang_velocity = (
            data_gps.urdf_orientation(iteration, link_i),
            data_gps.com_lin_velocity(iteration, link_i),
            data_gps.com_ang_velocity(iteration, link_i)
        )
        link_orientation_inv = np.array(
            pybullet.getMatrixFromQuaternion(ori)
        ).reshape([3, 3]).T
        link_velocity = np.dot(link_orientation_inv, lin_velocity)
        link_angular_velocity = np.dot(link_orientation_inv, ang_velocity)
        # Data
        data_hydrodynamics[iteration, link_i, :3] = (
            force_coefficients*link_velocity
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
        # if link_i == 11:
            # dynamics = pybullet.getDynamicsInfo(model, -1)
            # pos = np.array(dynamics[3])
            # base = data_gps.com_position(iteration, link_i) + np.dot(
            #     np.array(
            #         pybullet.getMatrixFromQuaternion(ori)
            #     ).reshape([3, 3]),
            #     -pos
            # )
        joint = np.array(data_gps.urdf_position(iteration, link_i))
        if link_i == 11:
            print("RBP position: {}".format(np.array(joint)))
        joint_ori = np.array(data_gps.urdf_orientation(iteration, link_i))
        com_ori = np.array(data_gps.com_orientation(iteration, link_i))
        ori_joint = np.array(
            pybullet.getMatrixFromQuaternion(joint_ori)
        ).reshape([3, 3])
        ori_com = np.array(
            pybullet.getMatrixFromQuaternion(com_ori)
        ).reshape([3, 3])
        ori = np.dot(ori_joint, ori_com)
        a = 0.05
        offset_x = np.dot(ori_joint, np.array([a, 0, 0]))
        offset_y = np.dot(ori_joint, np.array([0, a, 0]))
        offset_z = np.dot(ori_joint, np.array([0, 0, a]))
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
            # com = data_gps.com_position(iteration, link_i)
            # pybullet.addUserDebugLine(
            #     com,
            #     com + np.array([0, 0, 0.1]),
            #     lineColorRGB=[0, 0, 1],
            #     lifeTime=1
            # )
