"""Animat options"""


from enum import IntEnum
from farms_data.options import Options
# from farms_bullet.model.control import ControlType


class SpawnLoader(IntEnum):
    """Spawn loader"""
    FARMS = 0
    PYBULLET = 1


class ControlOptions(Options):
    """Control options"""

    def __init__(self, **kwargs):
        super(ControlOptions, self).__init__()
        sensors = kwargs.pop('sensors')
        self.sensors = (
            sensors
            if isinstance(sensors, SensorsOptions)
            else SensorsOptions(**kwargs.pop('sensors'))
        )
        joints = kwargs.pop('joints')
        self.joints = (
            joints
            if all([
                isinstance(joint, JointControlOptions)
                for joint in joints
            ])
            else [
                JointControlOptions(**joint)
                for joint in joints
            ]
        )
        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))

    @staticmethod
    def options_from_kwargs(kwargs):
        """Options from kwargs"""
        options = {}
        options['sensors'] = kwargs.pop(
            'sensors',
            SensorsOptions.from_options(kwargs).to_dict()
        )
        options['joints'] = kwargs.pop('joints', [])
        return options

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        return cls(**cls.options_from_kwargs(kwargs))

    def joints_max_torque(self):
        """Joints max torques"""
        return [joint.max_torque for joint in self.joints]


class JointControlOptions(Options):
    """Joint options"""

    def __init__(self, **kwargs):
        super(JointControlOptions, self).__init__()
        self.joint = kwargs.pop('joint')
        self.control_type = kwargs.pop('control_type')
        self.max_torque = kwargs.pop('max_torque')
        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))


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
