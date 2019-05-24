"""Control"""

import numpy as np

from ...controllers.control import (
    ModelController,
    SineControl,
    ControlPDF,
    JointController
)

# from .animat_options import SalamanderControlOptions
from .network import SalamanderNetworkODE


class SalamanderController(ModelController):
    """ModelController"""

    # @classmethod
    # def from_gait(cls, model, joints, gait, iterations, timestep, **kwargs):
    #     """Salamander controller from gait"""
    #     return cls.from_options(
    #         model=model,
    #         joints=joints,
    #         options=SalamanderControlOptions.default(**kwargs),
    #         iterations=iterations,
    #         timestep=timestep
    #     )

    # def update_gait(self, gait, joints, timestep):
    #     """Update gait"""
    #     controllers_body, controllers_legs = (
    #         SalamanderController.joints_controllers(
    #             joints=joints,
    #             options=SalamanderControlOptions.from_gait(
    #                 gait=gait,
    #                 frequency=self._frequency,
    #                 body_offset=self._body_offset
    #             )
    #         )
    #     )
    #     self.controllers = controllers_body + controllers_legs
    #     self.network.update_gait(gait)

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
        # frequency = options["frequency"]
        # amplitudes = np.linspace(
        #     options["body_amplitude_0"],
        #     options["body_amplitude_1"],
        #     n_body_joints
        # )
        # joint_controllers_body = [
        #     JointController(
        #         joint=joints["joint_link_body_{}".format(joint_i)],
        #         sine=SineControl(
        #             amplitude=amplitudes[joint_i] + (
        #                 options["body_stand_amplitude"]*np.sin(
        #                     2*np.pi*joint_i/n_body_joints
        #                     - options["body_stand_shift"]
        #                 )
        #             ),
        #             frequency=frequency,
        #             offset=0
        #         ),
        #         pdf=(
        #             ControlPDF(
        #                 p=options["body_p"],
        #                 d=options["body_d"],
        #                 f=options["body_f"]
        #             )
        #         ),
        #         is_body=True
        #     )
        #     for joint_i in range(n_body_joints)
        # ]
        # n_dof_legs = 4
        # joint_controllers_legs = [
        #     JointController(
        #         joint=joints["joint_link_leg_{}_{}_{}".format(
        #             leg_i,
        #             side,
        #             joint_i
        #         )],
        #         sine=SineControl(
        #             amplitude=options["leg_{}_amplitude".format(joint_i)],
        #             frequency=frequency,
        #             offset=options["leg_{}_offset".format(joint_i)]
        #         ),
        #         pdf=ControlPDF(
        #             p=options["legs_p"],
        #             d=options["legs_d"],
        #             f=options["legs_f"]
        #         )
        #     )
        #     for leg_i in range(2)
        #     for side_i, side in enumerate(["L", "R"])
        #     for joint_i in range(n_dof_legs)
        # ]
        # return joint_controllers_body, joint_controllers_legs
        joint_controllers_body = [
            JointController(
                joint=joints["joint_link_body_{}".format(joint_i)],
                sine=SineControl(
                    amplitude=0,
                    frequency=0,
                    offset=0
                ),
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
                sine=SineControl(
                    amplitude=0,
                    frequency=0,
                    offset=0
                ),
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
