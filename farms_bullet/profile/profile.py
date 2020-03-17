"""Profile"""

import farms_pylog as pylog


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
        pylog.debug("\n".join((
            "Time to simulate {} [s]: {} [s]".format(
                self.sim_duration,
                self.sim_time,
            ),
            "  Plugin: {} [s]".format(self.plugin_time),
            "  Bullet physics: {} [s]".format(self.physics_time),
            "  Controller: {} [s]".format(self.ctrl_time),
            "  Sensors: {} [s]".format(self.sensors_time),
            "  Logging: {} [s]".format(self.log_time),
            "  Camera: {} [s]".format(self.camera_time),
            "  Wait real-time: {} [s]".format(self.waitrt_time),
            "  Sum: {} [s]".format(
                self.plugin_time
                + self.physics_time
                + self.ctrl_time
                + self.sensors_time
                + self.log_time
                + self.camera_time
                + self.waitrt_time
            ),
        )))
