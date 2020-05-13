"""Interface"""

import pybullet
import farms_pylog as pylog
from .camera import UserCamera, CameraRecord


class Interfaces:
    """Interfaces (GUI, camera, video)"""

    def __init__(
            self,
            camera=None,
            user_params=None,
            video=None,
            camera_skips=1
    ):
        super(Interfaces, self).__init__()
        self.camera = camera
        self.user_params = user_params
        self.video = video
        self.camera_skips = camera_skips

    def init_camera(self, target_identity, timestep, **kwargs):
        """Initialise camera"""
        # Camera
        self.camera = UserCamera(
            target_identity=target_identity,
            yaw=0,
            yaw_speed=(
                360/10*self.camera_skips
                if kwargs.pop('rotating_camera', False)
                else 0
            ),
            pitch=-89 if kwargs.pop('top_camera', False) else -45,
            distance=1,
            timestep=timestep
        )

    def init_video(self, target_identity, simulation_options, **kwargs):
        """Initialise video recording"""
        self.video = CameraRecord(
            timestep=simulation_options.timestep,
            target_identity=target_identity,
            n_iterations=simulation_options.n_iterations,
            fps=simulation_options.fps,
            pitch=kwargs.pop('pitch', simulation_options.video_pitch),
            yaw=kwargs.pop('yaw', simulation_options.video_yaw),
            yaw_speed=(
                1000
                if kwargs.pop('rotating_camera', False)
                else 0
            ),
            motion_filter=kwargs.pop('motion_filter', 1e-1),
            distance=simulation_options.video_distance,
        )
        assert not kwargs, kwargs

    def init_debug(self, simulation_options):
        """Initialise debug"""
        # User parameters
        self.user_params = UserParameters(simulation_options)


class DebugParameter:
    """DebugParameter"""

    def __init__(self, name, val, val_min, val_max):
        super(DebugParameter, self).__init__()
        self.name = name
        self.value = val
        self.val_min = val_min
        self.val_max = val_max
        self._handler = None
        self.changed = False
        self.add(self.value)

    def add(self, value):
        """Add parameter"""
        if self._handler is None:
            self._handler = pybullet.addUserDebugParameter(
                paramName=self.name,
                rangeMin=self.val_min,
                rangeMax=self.val_max,
                startValue=value
            )
        else:
            raise Exception(
                'Handler for parameter \'{}\' is already used'.format(
                    self.name
                )
            )

    def remove(self):
        """Remove parameter"""
        pybullet.removeUserDebugItem(self._handler)

    def get_value(self):
        """Current value"""
        return pybullet.readUserDebugParameter(self._handler)

    def update(self):
        """Update"""
        previous_value = self.value
        self.value = self.get_value()
        self.changed = (self.value != previous_value)
        if self.changed:
            pylog.debug('{} changed ({} -> {})'.format(
                self.name,
                previous_value,
                self.value
            ))


class ParameterPlay(DebugParameter):
    """Play/pause parameter

    self.value:          Boolean
    self.previous_value: Keeps track of the last measured value
    self.get_value():    Returns the number of times play/plause button
                         has been pressed
    """

    def __init__(self, initial_value=True):
        self.previous_value = 0
        super(ParameterPlay, self).__init__(
            'Play/Pause', self.previous_value, 0, -1
        )
        self.value = initial_value

    def update(self):
        """Update"""
        value = self.get_value()
        if value != self.previous_value:
            self.value = not self.value
            self.previous_value = value


class UserParameters(dict):
    """Parameters control"""

    def __init__(self, options):
        super(UserParameters, self).__init__()
        self['play'] = ParameterPlay(initial_value=options.play)
        self['rtl'] = DebugParameter('Real-time limiter', 1, 1e-3, 3)
        self['zoom'] = DebugParameter('Zoom', 1, 0, 1)

    def update(self):
        """Update parameters"""
        for parameter in self:
            self[parameter].update()

    def play(self):
        """Play"""
        return self['play']

    def rtl(self):
        """Real-time limiter"""
        return self['rtl']

    def zoom(self):
        """Camera zoom"""
        return self['zoom']
