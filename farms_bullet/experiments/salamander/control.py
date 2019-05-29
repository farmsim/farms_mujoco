"""Control"""

from ...controllers.control import AnimatController
from .network import SalamanderNetworkODE


class SalamanderController(AnimatController):
    """AnimatController"""

    @classmethod
    def from_options(cls, model, options, iterations, timestep):
        """Salamander controller from options"""
        return cls(
            model=model,
            network=SalamanderNetworkODE.from_options(
                options,
                iterations,
                timestep
            )
        )
