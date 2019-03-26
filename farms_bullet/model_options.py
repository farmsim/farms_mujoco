"""Model options"""

class ModelOptions(dict):
    """Simulation options"""

    def __init__(self, **kwargs):
        super(ModelOptions, self).__init__()
        self["gait"] = kwargs.pop("gait", "walking")
        self["frequency"] = kwargs.pop("frequency", 1)
        self["body_stand_amplitude"] = kwargs.pop("body_stand_amplitude", 0.2)

    @property
    def frequency(self):
        """Model frequency"""
        return self["frequency"]

    @property
    def body_stand_amplitude(self):
        """Model body amplitude"""
        return self["body_stand_amplitude"]

    @property
    def gait(self):
        """Model gait"""
        return self["gait"]
