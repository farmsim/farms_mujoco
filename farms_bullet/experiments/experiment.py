"""Experiment"""

import numpy as np

class Experiment:
    """Experiment

    All experiment related code (Pure simulation). This does not include
    GUI/headless mode, simulation physics, timesteps handling, etc.

    TODO: The previous simulation class and experiment class should be merged
    into one to facilitate experiments design. As such, the experiment should
    contain simulation properties, an arena and animat(s). It would also contain
    the methods for (re)spawning, running and logging the experiment.

    """

    def __init__(self, animat, arena, timestep, n_iterations):
        super(Experiment, self).__init__()
        self.animat = animat
        self.arena = arena
        self.timestep = timestep
        self.n_iterations = n_iterations

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

    def postprocess(self, iteration, **kwargs):
        """Plot after simulation"""
        times = np.arange(
            0,
            self.timestep*self.n_iterations,
            self.timestep
        )[:iteration]

        plot = kwargs.pop("plot", None)
        if plot:
            self.logger.plot_all(times)

        log_path = kwargs.pop("log_path", None)
        if log_path:
            log_extension = kwargs.pop("log_extension", None)
            self.logger.log_all(
                times,
                folder=log_path,
                extension=log_extension
            )

        # Record video
        record = kwargs.pop("record", None)
        if record:
            self.camera_record.save("video.avi")

    def end(self, **kwargs):
        """delete"""
