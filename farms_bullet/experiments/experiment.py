"""Experiment"""

class Experiment:
    """Experiment"""

    def __init__(self, animat, arena, timestep, n_iterations):
        super(Experiment, self).__init__()
        self.animat = animat
        self.arena = arena
        self.timestep = timestep
        self.n_iterations = n_iterations
        self.logger = None

    def elements(self):
        """Elements in experiment"""
        return [self.animat, self.arena]

    def _spawn(self):
        """Spawn"""
        for element in self.elements():
            element.spawn()

    def spawn(self):
        """Spawn"""
        self._spawn()

    def step(self):
        """Step"""
        for element in self.elements():
            element.step()

    def log(self):
        """Step"""
        for element in self.elements():
            element.log()

    def end(self):
        """delete"""
