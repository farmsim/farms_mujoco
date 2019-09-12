"""Control"""

from ...controllers.control import AnimatController
from .network import AmphibiousNetworkODE


class AmphibiousController(AnimatController):
    """AnimatController"""

    @classmethod
    def from_data(cls, model, animat_options, animat_data, timestep, joints_order, units):
        """Amphibious controller from options"""
        return cls(
            model=model,
            network=AmphibiousNetworkODE(animat_options, animat_data, timestep),
            joints_order=joints_order,
            units=units
        )
