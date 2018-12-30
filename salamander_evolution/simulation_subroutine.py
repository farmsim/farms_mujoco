import importlib.util
import numpy as np
from salamander_generation import generate_walking

#foo.run_island() command to run for starting the simulation
#spec_run_island = importlib.util.spec_from_file_location("gazebo_island.py",
#         "/home/blaise/Desktop/salamander_sim_students/salamander_experiments/openmpi/gazebo_island.py")
#foo_run_island = importlib.util.module_from_spec(spec_run_island)
#spec_run_island.loader.exec_module(spec_run_island)


spec_fitness = importlib.util.spec_from_file_location("compute_fitness.py",
         "/home/blaise/Desktop/salamander_sim_students/salamander_experiments/openmpi/compute_fitness.py")
foo_fitness = importlib.util.module_from_spec(spec_fitness)
spec_fitness.loader.exec_module(spec_fitness)




class evol_problem:
    def __init__(self, dim, link, path):
        self.dim = dim
        self.link = link
        self.path = path

    def fitness(self):
        generate_walking()
        foo_gen_all.run_island()
        #path ".gazebo/models/salamander_new"
        foo_fitness.compute_fitness(self.path, self.link)

        obj_func = np.array([])
        return obj_func

    def get_bounds(self):
        return