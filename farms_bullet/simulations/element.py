"""Simulation element"""


class SimulationElement:
    """Documentation for SimulationElement"""

    def __init__(self):
        super(SimulationElement, self).__init__()
        self._identity = None

    @property
    def identity(self):
        """Element identity"""
        return self._identity

    @staticmethod
    def spawn():
        """Spawn"""

    @staticmethod
    def step():
        """Step"""

    @staticmethod
    def log():
        """Log"""

    @staticmethod
    def save_logs():
        """Save logs"""

    @staticmethod
    def plot():
        """Plot"""

    @staticmethod
    def reset():
        """Reset"""

    @staticmethod
    def delete():
        """Delete"""
