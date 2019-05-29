"""Control"""

import numpy as np

from ...controllers.control import (
    ModelController,
    SineControl,
    ControlPDF,
    JointController
)

from .network import SalamanderNetworkODE


class SalamanderController(ModelController):
    """ModelController"""

    @classmethod
    def from_options(cls, model, joints, options, iterations, timestep):
        """Salamander controller from options"""
        joint_controllers_body, joint_controllers_legs = (
            cls.joints_controllers(joints, options)
        )
        return cls(
            model=model,
            network=SalamanderNetworkODE.from_options(
                options,
                iterations,
                timestep
            ),
            joints_controllers=joint_controllers_body + joint_controllers_legs
        )

    @staticmethod
    def joints_controllers(joints, options):
        """Controllers"""
        n_body_joints = options.morphology.n_joints_body
        joint_controllers_body = [
            JointController(
                joint=joints["joint_link_body_{}".format(joint_i)],
                pdf=(
                    ControlPDF(
                        p=options.control.joints_controllers.body_p,
                        d=options.control.joints_controllers.body_d,
                        f=options.control.joints_controllers.body_f
                    )
                ),
                is_body=True
            )
            for joint_i in range(n_body_joints)
        ]
        joint_controllers_legs = [
            JointController(
                joint=joints["joint_link_leg_{}_{}_{}".format(
                    leg_i,
                    side,
                    joint_i
                )],
                pdf=ControlPDF(
                    p=options.control.joints_controllers.legs_p,
                    d=options.control.joints_controllers.legs_d,
                    f=options.control.joints_controllers.legs_f
                )
            )
            for leg_i in range(2)
            for side_i, side in enumerate(["L", "R"])
            for joint_i in range(options.morphology.n_dof_legs)
        ]
        return joint_controllers_body, joint_controllers_legs
