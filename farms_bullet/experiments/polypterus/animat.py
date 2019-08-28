"""Polypterus"""

import os
import numpy as np
from ...animats.amphibious.animat import Amphibious


class Polypterus(Amphibious):
    """Polypterus"""

    def __init__(self, options, timestep, iterations, units):
        options.morphology.n_dof_legs = 4
        options.morphology.n_legs = 2
        options.morphology.leg_length = 0.02
        options.morphology.leg_radius = 0.01
        options.control.network.connectivity.body_phase_bias = (
            np.pi/options.morphology.n_joints_body
        )
        use_directory = False
        options.morphology.mesh_directory = (
            "/{}/meshes".format(
                os.path.dirname(os.path.realpath(__file__))
            ) if use_directory else ""
        )
        super(Polypterus, self).__init__(options, timestep, iterations, units)
