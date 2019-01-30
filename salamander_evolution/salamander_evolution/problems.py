"""Evolution problems"""

import salamander_pyrun as sr
# from salamander_results import extract_logs
from salamander_results.extract import extract_positions
from salamander_generation import generate_walking, ControlParameters

import pygmo as pg
import numpy as np


class ProblemWalkingFrequency:
    """Walking frequency problem"""

    def __init__(self, link_name):
        super(ProblemWalkingFrequency, self).__init__()
        self._dim = 1
        self.link_name = link_name

    def fitness(self, decision_vector):
        """Fitness function"""
        print("Decision vector:\n{}".format(decision_vector))
        freq = decision_vector[0]
        name = "salamander_{:.3f}".format(float(freq)).replace(".", "d")
        control_data = ControlParameters(gait="walking", frequency=float(freq))
        generate_walking(name=name, control_parameters=control_data)

        #Running the simulation
        world_path = "/.gazebo/models/{}/world.world".format(name)
        sr.run_simulation(world_path)

        #computing the fitness function & extraction of the simulation logs
        path = ".gazebo/models/{}".format(name)
        indiv_pos = extract_positions(path, self.link_name)
        obj_func = max([
            np.linalg.norm(indiv_pos[i])
            for i in np.arange(0, len(indiv_pos))
        ])
        return np.array([-obj_func])

    def get_bounds(self):
        """Get bounds"""
        return ([0] * self._dim, [2] * self._dim)

    @staticmethod
    def get_name():
        """Get name"""
        return "Evolution of the salamander (Frequency evolution)"

    def get_extra_info(self):
        """Get extra info"""
        return "\tDimensions: " + str(self._dim)
