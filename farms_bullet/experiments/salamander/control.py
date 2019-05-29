"""Control"""

from ...controllers.control import ModelController
from .network import SalamanderNetworkODE


class SalamanderController(ModelController):
    """ModelController"""

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
