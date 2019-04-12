"""Profile"""


class SimulationProfiler:
    """Simulation profiler"""

    def __init__(self, sim_duration):
        super(SimulationProfiler, self).__init__()
        self.sim_duration = sim_duration
        self.plugin_time = 0
        self.sim_time = 0
        self.physics_time = 0
        self.ctrl_time = 0
        self.sensors_time = 0
        self.log_time = 0
        self.camera_time = 0
        self.waitrt_time = 0

    def reset(self):
        """Reset"""
        self.plugin_time = 0
        self.sim_time = 0
        self.physics_time = 0
        self.ctrl_time = 0
        self.sensors_time = 0
        self.log_time = 0
        self.camera_time = 0
        self.waitrt_time = 0

    def total_time(self):
        """Total time"""
        return (
            self.plugin_time
            + self.physics_time
            + self.ctrl_time
            + self.sensors_time
            + self.log_time
            + self.camera_time
            + self.waitrt_time
        )

    def print_times(self):
        """Print times"""
        print("Time to simulate {} [s]: {} [s]".format(
            self.sim_duration,
            self.sim_time,
        ))
        print("  Plugin: {} [s]".format(self.plugin_time))
        print("  Bullet physics: {} [s]".format(self.physics_time))
        print("  Controller: {} [s]".format(self.ctrl_time))
        print("  Sensors: {} [s]".format(self.sensors_time))
        print("  Logging: {} [s]".format(self.log_time))
        print("  Camera: {} [s]".format(self.camera_time))
        print("  Wait real-time: {} [s]".format(self.waitrt_time))
        print("  Sum: {} [s]".format(
            self.plugin_time
            + self.physics_time
            + self.ctrl_time
            + self.sensors_time
            + self.log_time
            + self.camera_time
            + self.waitrt_time
        ))
