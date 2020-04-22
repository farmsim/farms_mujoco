"""Animat data"""

import numpy as np
cimport numpy as np


cdef class SensorsDataCy:
    """SensorsData"""

    def __init__(
            self,
            ContactsArrayCy contacts=None,
            ProprioceptionArrayCy proprioception=None,
            GpsArrayCy gps=None,
            HydrodynamicsArrayCy hydrodynamics=None
    ):
        super(SensorsDataCy, self).__init__()
        self.contacts = contacts
        self.proprioception = proprioception
        self.gps = gps
        self.hydrodynamics = hydrodynamics
