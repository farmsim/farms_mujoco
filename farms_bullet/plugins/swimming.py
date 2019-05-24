"""Swimming"""

import numpy as np
import pybullet


def viscous_swimming(model, links):
    """Viscous swimming"""
    # Swimming
    forces_torques = np.zeros([2, 10, 3])
    for link_i in range(1, 11):
        link_state = pybullet.getLinkState(
            model,
            links["link_body_{}".format(link_i)],
            computeLinkVelocity=1,
            computeForwardKinematics=0
        )
        link_orientation_inv = np.linalg.inv(np.array(
            pybullet.getMatrixFromQuaternion(link_state[5])
        ).reshape([3, 3]))
        link_velocity = np.dot(link_orientation_inv, link_state[6])
        link_angular_velocity = np.dot(link_orientation_inv, link_state[7])
        forces_torques[0, link_i-1, :] = (
            np.array([-1e-1, -1e0, -1e0])*link_velocity
        )
        pybullet.applyExternalForce(
            model,
            links["link_body_{}".format(link_i)],
            forceObj=forces_torques[0, link_i-1, :],
            posObj=[0, 0, 0],
            flags=pybullet.LINK_FRAME
        )
        forces_torques[1, link_i-1, :] = (
            np.array([-1e-2, -1e-2, -1e-2])*link_angular_velocity
        )
        pybullet.applyExternalTorque(
            model,
            links["link_body_{}".format(link_i+1)],
            torqueObj=forces_torques[1, link_i-1, :],
            flags=pybullet.LINK_FRAME
        )
    return forces_torques
