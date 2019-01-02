
import numpy as np
#from salamander_pyrun import salamander_pyrun
import salamander_pyrun as sr
from salamander_results import extract_logs
from salamander_results import plot
from salamander_generation import generate_walking, ControlParameters


class evol_problem:
    def __init__(self, dim, link_name, path):
        self.dim = dim
        self.link_name = link_name
        self.path = path
        self.name = ""

    def fitness(self, x):
        freq = x
        self.name = "salamander_{:.2f}".format(float(freq)).replace(".", "d")
        control_data = ControlParameters(gait="walking", frequency=float(freq))
        generate_walking(name=self.name, control_parameters=control_data)

        #Running the simulation
        world_path = "/.gazebo/models/{}/world.world".format(self.name)
        sr.run_simulation(world_path)

        #computing the fitness function & extraction of the simulation logs
        self.path = ".gazebo/models/{}".format(self.name)
        indiv_pos = plot.positions(self.path, self.link_name)
        obj_func = max([np.linalg.norm(indiv_pos[i]) for i in np.arange(0,len(indiv_pos))])
        return np.array([obj_func])

    def get_bounds(self):
        return ([0] * self.dim, [2] * self.dim)

    def get_name(self):
        return "Evolution of the salamdar (Frequency evolution)"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)