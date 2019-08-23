"""Control"""

from ...controllers.control import AnimatController
from .network import SalamanderNetworkODE


class SalamanderController(AnimatController):
    """AnimatController"""

    @classmethod
    def from_data(cls, model, animat_options, animat_data, timestep, joints_order, units):
        """Salamander controller from options"""
        return cls(
            model=model,
            network=SalamanderNetworkODE(animat_options, animat_data, timestep),
            joints_order=joints_order,
            units=units
        )
