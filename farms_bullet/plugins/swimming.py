"""Swimming"""

import numpy as np
import pybullet


def viscous_swimming(iteration, data, model, links, plot=False):
    """Viscous swimming"""
    # Swimming
    force_coefficients = np.array([-1e-1, -1e0, -1e0])
    torque_coefficients = np.array([-1e-2, -1e-2, -1e-2])
    for link_i in range(12):
        # Collect data
        link = links["link_body_{}".format(link_i)]
        if link_i == 0:
            # Base link
            pos, ori = pybullet.getBasePositionAndOrientation(model)
            lin_velocity, ang_velocity = pybullet.getBaseVelocity(model)
        else:
            # Children links
            link_state = pybullet.getLinkState(
                model,
                link,
                computeLinkVelocity=1,
                computeForwardKinematics=0
            )
            pos, ori, lin_velocity, ang_velocity = (
                link_state[0],
                link_state[5],
                link_state[6],
                link_state[7]
            )
        link_orientation_inv = np.linalg.inv(np.array(
            pybullet.getMatrixFromQuaternion(ori)
        ).reshape([3, 3]))
        link_velocity = np.dot(link_orientation_inv, lin_velocity)
        link_angular_velocity = np.dot(link_orientation_inv, ang_velocity)
        # Data
        data[iteration, link_i, :3] = force_coefficients*link_velocity
        data[iteration, link_i, 3:6] = torque_coefficients*link_angular_velocity
        # Forces
        pybullet.applyExternalForce(
            model,
            link,
            forceObj=data[iteration, link_i, :3],
            posObj=[0, 0, 0],
            flags=pybullet.LINK_FRAME
        )
        pybullet.applyExternalTorque(
            model,
            link,
            torqueObj=data[iteration, link_i, 3:6],
            flags=pybullet.LINK_FRAME
        )
        if plot:
            # Debug
            pos = [list(pos), list(pos)]
            pos[0][2] = 0
            pos[1][2] = 1
            line = pybullet.addUserDebugLine(
                lineFromXYZ=pos[0],
                lineToXYZ=pos[1],
                lineColorRGB=[0.1, 0.5, link_i/12],
                lineWidth=3,
                lifeTime=0.1
            )
