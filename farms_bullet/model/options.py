"""Animat options"""


from enum import IntEnum
from farms_data.options import Options


class SpawnLoader(IntEnum):
    """Spawn loader"""
    FARMS = 0
    PYBULLET = 1


class MorphologyOptions(Options):
    """ morphology options"""

    def __init__(self, **kwargs):
        super(MorphologyOptions, self).__init__()
        links = kwargs.pop('links')
        self.links = (
            links
            if all([isinstance(link, LinkOptions) for link in links])
            else [LinkOptions(**link) for link in kwargs.pop('links')]
        )
        self.self_collisions = kwargs.pop('self_collisions')
        joints = kwargs.pop('joints')
        self.joints = (
            joints
            if all([isinstance(joint, JointOptions) for joint in joints])
            else [JointOptions(**joint) for joint in joints]
        )
        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))

    def links_names(self):
        """Links names"""
        return [link.name for link in self.links]

    def joints_names(self):
        """Joints names"""
        return [joint.name for joint in self.joints]

    def n_joints(self):
        """Number of joints"""
        return len(self.joints)

    def n_links(self):
        """Number of links"""
        return len(self.links)


class LinkOptions(Options):
    """Link options

    The Pybullet dynamics represent the input arguments called with
    pybullet.changeDynamics(...).
    """

    def __init__(self, **kwargs):
        super(LinkOptions, self).__init__()
        self.name = kwargs.pop('name')
        self.collisions = kwargs.pop('collisions')
        self.mass_multiplier = kwargs.pop('mass_multiplier')
        self.pybullet_dynamics = kwargs.pop('pybullet_dynamics', {})
        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))


class JointOptions(Options):
    """Joint options

    The Pybullet dynamics represent the input arguments called with
    pybullet.changeDynamics(...). The appropriate link is called for it.
    """

    def __init__(self, **kwargs):
        super(JointOptions, self).__init__()
        self.name = kwargs.pop('name')
        self.initial_position = kwargs.pop('initial_position')
        self.initial_velocity = kwargs.pop('initial_velocity')
        self.pybullet_dynamics = kwargs.pop('pybullet_dynamics', {})
        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))


class SpawnOptions(Options):
    """Spawn options"""

    def __init__(self, **kwargs):
        super(SpawnOptions, self).__init__()
        self.loader = kwargs.pop('loader')
        self.position = kwargs.pop('position')
        self.orientation = kwargs.pop('orientation')
        self.velocity_lin = kwargs.pop('velocity_lin')
        self.velocity_ang = kwargs.pop('velocity_ang')
        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        # Loader
        options['loader'] = kwargs.pop('spawn_loader', SpawnLoader.PYBULLET)
        # Position in [m]
        options['position'] = kwargs.pop('spawn_position', [0, 0, 0.1])
        # Orientation in [rad] (Euler angles)
        options['orientation'] = kwargs.pop('spawn_orientation', [0, 0, 0])
        # Linear velocity in [m/s]
        options['velocity_lin'] = kwargs.pop('spawn_velocity_lin', [0, 0, 0])
        # Angular velocity in [rad/s] (Euler angles)
        options['velocity_ang'] = kwargs.pop('spawn_velocity_ang', [0, 0, 0])
        return cls(**options)


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
            if all([isinstance(joint, JointControlOptions) for joint in joints])
            else [JointControlOptions(**joint) for joint in joints]
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
