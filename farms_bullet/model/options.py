"""Animat options"""

from farms_data.options import Options


class SensorsOptions(Options):
    """Sensors options"""

    def __init__(self, **kwargs):
        super(SensorsOptions, self).__init__()
        self.gps = kwargs.pop('gps')
        self.joints = kwargs.pop('joints')
        self.contacts = kwargs.pop('contacts')
        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))

    @staticmethod
    def options_from_kwargs(kwargs):
        """Options from kwargs"""
        options = {}
        options['gps'] = kwargs.pop('sens_gps', None)
        options['joints'] = kwargs.pop('sens_joints', None)
        options['contacts'] = kwargs.pop('sens_contacts', None)
        return options

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        return cls(**cls.options_from_kwargs(kwargs))
