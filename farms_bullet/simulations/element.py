"""Simulation element"""

import pybullet


class SimulationElement:
    """Documentation for SimulationElement"""

    def __init__(self, identity=None):
        super(SimulationElement, self).__init__()
        self._identity = identity

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

    # @classmethod
    # def from_sdf(cls, sdf, options=None, sdf_options=None):
    #     """Model from SDF"""
    #     if options is None:
    #         options = {}
    #     if sdf_options is None:
    #         sdf_options = {}
    #     identity = pybullet.loadSDF(sdf, **sdf_options)[0]
    #     return cls(identity, **options)

    # @classmethod
    # def from_urdf(cls, urdf, options=None, urdf_options=None):
    #     """Model from SDF"""
    #     if options is None:
    #         options = {}
    #     if sdf_options is None:
    #         sdf_options = {}
    #     identity = pybullet.loadURDF(urdf, urdf_options)
    #     return cls(identity, **options)

    @staticmethod
    def from_sdf(sdf, **kwargs):
        """Model from SDF"""
        return pybullet.loadSDF(sdf, **kwargs)[0]

    @staticmethod
    def from_urdf(urdf, **kwargs):
        """Model from SDF"""
        return pybullet.loadURDF(urdf, **kwargs)
