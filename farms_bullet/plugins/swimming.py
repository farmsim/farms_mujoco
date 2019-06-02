"""Swimming"""

import numpy as np
import pybullet


def get_gps(iteration, data, model, links, _links):
    """Get GPS of links"""
    for link_i, link in _links:
        # Collect data
        if link_i == 0:
            # Base link
            pos, ori = pybullet.getBasePositionAndOrientation(model)
            lin_velocity, ang_velocity = pybullet.getBaseVelocity(model)
        else:
            # Children links
            link_state = pybullet.getLinkState(
                model,
                links[link],
                computeLinkVelocity=1,
                computeForwardKinematics=0
            )
            pos, ori, lin_velocity, ang_velocity = (
                link_state[0],
                link_state[5],
                link_state[6],
                link_state[7]
            )
        data[iteration, link_i, :3] = np.array(pos)
        data[iteration, link_i, 3:7] = np.array(ori)
        data[iteration, link_i, 7:10] = np.array(lin_velocity)
        data[iteration, link_i, 10:13] = np.array(ang_velocity)


def viscous_swimming(iteration, data_gps, data_hydrodynamics, model, links, _links):
    """Viscous swimming"""
    # Swimming
    force_coefficients = np.array([-1e-1, -1e0, -1e0])
    torque_coefficients = np.array([-1e-2, -1e-2, -1e-2])
    for link_i, link in _links:
        ori, lin_velocity, ang_velocity = (
            data_gps[iteration, link_i, 3:7],
            data_gps[iteration, link_i, 7:10],
            data_gps[iteration, link_i, 10:13]
        )
        link_orientation_inv = np.linalg.inv(np.array(
            pybullet.getMatrixFromQuaternion(ori)
        ).reshape([3, 3]))
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
            links[link],
            forceObj=data_hydrodynamics[iteration, link_i, :3],
            posObj=[0, 0, 0],
            flags=pybullet.LINK_FRAME
        )
        pybullet.applyExternalTorque(
            model,
            links[link],
            torqueObj=data_hydrodynamics[iteration, link_i, 3:6],
            flags=pybullet.LINK_FRAME
        )
