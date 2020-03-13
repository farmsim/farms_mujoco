"""Salamander"""

import os

from farms_models.utils import get_sdf_path
from ...animats.amphibious.animat import Amphibious


class Salamander(Amphibious):
    """Salamander"""

    def __init__(self, options, timestep, iterations, units):
        options.morphology.n_dof_legs = 4
        options.morphology.n_legs = 4
        use_directory = True
        options.morphology.mesh_directory = (
            "{}/meshes".format(
                os.path.dirname(os.path.realpath(__file__))
            ) if use_directory else ""
        )
        super(Salamander, self).__init__(
            options,
            timestep,
            iterations,
            units,
            sdf=get_sdf_path(name='salamander', version='v1')
        )
