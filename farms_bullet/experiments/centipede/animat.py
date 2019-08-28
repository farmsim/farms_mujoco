"""Centipede"""

import numpy as np
from ...animats.amphibious.animat import Amphibious

class Centipede(Amphibious):
    """Centipede"""

    def __init__(self, options, timestep, iterations, units):
        options.morphology.n_joints_body = 15
        options.morphology.n_dof_legs = 4
        options.morphology.n_legs = 2*(options.morphology.n_links_body()-2)
        options.morphology.legs_parents = np.arange(options.morphology.n_legs//2)
        options.morphology.mesh_directory = ""
        options.morphology.leg_length = 0.04
        options.morphology.leg_radius = 0.007
        options.control.network.connectivity.leg_phase_follow = (
            3*2*np.pi/options.morphology.n_legs
        )
        options.control.network.oscillators.set_body_stand_amplitude(0)
        options.control.network.connectivity.weight_osc_legs_opposite = 1e2
        options.control.network.connectivity.weight_osc_legs_following = 1e2
        options.control.network.connectivity.weight_sens_contact_i = 0
        options.control.network.connectivity.weight_sens_contact_e = 0
        options.control.network.connectivity.weight_sens_hydro_freq = 0
        options.control.network.connectivity.weight_sens_hydro_amp = 0
        options.control.network.update()
        super(Centipede, self).__init__(options, timestep, iterations, units)
