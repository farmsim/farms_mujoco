"""Algorithms for benchmarking"""

import numpy as np


class QuadraticFunction:
    """QuadraticFunction"""

    def __init__(self, dim, slow=False):
        super(QuadraticFunction, self).__init__()
        self._dim = dim
        self._slow = slow
        self._name = "Step-quadratic Function"

    @staticmethod
    def fitness_function(decision_vector):
        """Fitnesss"""
        fitness = np.linalg.norm(decision_vector - QuadraticFunction.best_known())
        return [fitness + (1 if fitness > 1 else 0)]

    def fitness(self, decision_vector):
        """Fitnesss"""
        if self._slow:
            time.sleep(0.5)
        return self.fitness_function(decision_vector)

    def get_name(self):
        """Get name"""
        return self._name

    @staticmethod
    def get_bounds():
        """Get bounds"""
        return ([-1, -1], [1, 1])

    @staticmethod
    def best_known():
        """Best known"""
        return np.array([0.5, 0.5])
