"""Quadruped"""

import os

from ...animats.amphibious.animat import Amphibious


class Quadruped(Amphibious):
    """Quadruped"""

    def __init__(self, options, timestep, iterations, units):
        options.morphology.n_joints_body = 6
        options.morphology.n_dof_legs = 4
        options.morphology.n_legs = 4
        use_directory = False
        options.morphology.mesh_directory = (
            "/{}/meshes".format(
                os.path.dirname(os.path.realpath(__file__))
            ) if use_directory else ""
        )
        super(Quadruped, self).__init__(options, timestep, iterations, units)
