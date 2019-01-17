""" Plot """

import os
from salamander_results import extract_positions
import numpy as np


def compute_fitness(path, link):
    """ Main """
    pos = extract_positions(path, link)
    distances = [np.linalg.norm(_pos) for _pos in pos[:, :2]]
    return max(distances)
