"""Model options"""

import numpy as np


class SalamanderOptions(dict):
    """Simulation options"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, **kwargs):
        super(SalamanderOptions, self).__init__()
        self.gait = kwargs.pop("gait", "walking")
        self.frequency = kwargs.pop("frequency", 1)
        self.body_amplitude_0 = kwargs.pop("body_amplitude_0", 0)
        self.body_amplitude_1 = kwargs.pop("body_amplitude_1", 0)
        self.body_stand_amplitude = kwargs.pop("body_stand_amplitude", 0.2)
        self.body_stand_shift = kwargs.pop("body_stand_shift", np.pi/4)

        # Legs
        self.leg_0_amplitude = kwargs.pop("leg_0_amplitude", 0.8)
        self.leg_0_offset = kwargs.pop("leg_0_offset", 0)

        self.leg_1_amplitude = kwargs.pop("leg_1_amplitude", np.pi/32)
        self.leg_1_offset = kwargs.pop("leg_1_offset", np.pi/32)

        self.leg_2_amplitude = kwargs.pop("leg_2_amplitude", np.pi/8)
        self.leg_2_offset = kwargs.pop("leg_2_offset", np.pi/8)
