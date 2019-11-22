"""Fish"""

import os

from ...animats.amphibious.animat import Amphibious


class Fish(Amphibious):
    """Fish"""

    def __init__(self, options, timestep, iterations, units, sdf_path):
        options.morphology.n_dof_legs = 0
        options.morphology.n_legs = 0
        use_directory = True
        directory = os.path.dirname(os.path.realpath(__file__))
        options.morphology.mesh_directory = (
            "{}/meshes".format(
                directory
            ) if use_directory else ""
        )
        super(Fish, self).__init__(
            options,
            timestep,
            iterations,
            units,
            sdf=sdf_path
        )

    @classmethod
    def from_fish_data(cls, fish_name, version, options, timestep, iterations, units):
        """From fish data"""
        directory = os.path.dirname(os.path.realpath(__file__))
        return cls(
            options,
            timestep,
            iterations,
            units,
            sdf_path=os.path.join(
                directory,
                fish_name,
                version,
                "sdf",
                "{}.sdf".format(fish_name)
            )
        )
