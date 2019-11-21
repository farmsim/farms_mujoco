"""Crescent_Gunnel"""

import os

from ...animats.amphibious.animat import Amphibious


class Crescent_Gunnel(Amphibious):
    """Crescent_Gunnel"""

    def __init__(self, options, timestep, iterations, units):
        options.morphology.n_dof_legs = 0
        options.morphology.n_legs = 0
        use_directory = True
        directory = os.path.dirname(os.path.realpath(__file__))
        options.morphology.mesh_directory = (
            "{}/meshes".format(
                directory
            ) if use_directory else ""
        )
        super(Crescent_Gunnel, self).__init__(
            options,
            timestep,
            iterations,
            units,
            sdf=os.path.join(directory, "sdf", "crescent_gunnel.sdf")
        )
