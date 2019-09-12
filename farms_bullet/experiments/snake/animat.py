"""Snake"""

from ...animats.amphibious.animat import Amphibious

class Snake(Amphibious):
    """Snake"""

    def __init__(self, options, timestep, iterations, units):
        options.morphology.n_dof_legs = 0
        options.morphology.n_legs = 0
        options.morphology.mesh_directory = ""
        super(Snake, self).__init__(options, timestep, iterations, units)
