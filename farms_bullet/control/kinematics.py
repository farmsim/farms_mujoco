"""Kinematics"""

import numpy as np
from scipy.interpolate import interp1d
from farms_bullet.model.control import ModelController, ControlType


def kinematics_interpolation(
        kinematics,
        sampling,
        timestep,
        n_iterations,
):
    """Kinematics interpolations"""
    data_duration = sampling*kinematics.shape[0]
    simulation_duration = timestep*n_iterations
    interp_x = np.arange(0, data_duration, sampling)
    interp_xn = np.arange(0, simulation_duration, timestep)
    assert data_duration >= simulation_duration, 'Data {} < {} Sim'.format(
        data_duration,
        simulation_duration
    )
    assert len(interp_x) == kinematics.shape[0]
    assert interp_x[-1] >= interp_xn[-1], 'Data[-1] {} < {} Sim[-1]'.format(
        interp_x[-1],
        interp_xn[-1]
    )
    return interp1d(
        interp_x,
        kinematics,
        axis=0
    )(interp_xn)


class KinematicsController(ModelController):
    """Amphibious kinematics"""

    def __init__(
            self,
            joints,
            kinematics,
            sampling,
            timestep,
            n_iterations,
            animat_data,
            max_torques,
    ):
        super(KinematicsController, self).__init__(
            joints=joints,
            control_types={joint: ControlType.POSITION for joint in joints},
            max_torques=max_torques,
        )
        assert kinematics.shape[1] == len(joints), (
            'Expected {} joints, but got {}'.format(
                len(joints),
                kinematics.shape[1],
            )
        )
        self.kinematics = kinematics_interpolation(
            kinematics=kinematics,
            sampling=sampling,
            timestep=timestep,
            n_iterations=n_iterations,
        )
        self.animat_data = animat_data

    def positions(self, iteration, time, timestep):
        """Postions"""
        return dict(zip(
            self.joints[ControlType.POSITION],
            self.kinematics[iteration],
        ))
