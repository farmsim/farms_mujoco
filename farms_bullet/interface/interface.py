"""Interface"""

import numpy as np
import pybullet
from .camera import UserCamera, CameraRecord
import farms_pylog as pylog


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
        """Init video"""
        # Video recording
        # self.video = CameraRecord(
        #     target_identity=target_identity,
        #     size=size,
        #     fps=kwargs.pop('fps', 40),
        #     yaw=kwargs.pop('yaw', 0),
        #     yaw_speed=360/10 if kwargs.pop('rotating_camera', False) else 0,
        #     pitch=-89 if kwargs.pop('top_camera', False) else -45,
        #     distance=1,
        #     timestep=timestep,
        #     motion_filter=1e-1
        # )
        skips = kwargs.pop('skips', 1)
        self.video = CameraRecord(
            target_identity=target_identity,
            size=simulation_options.n_iterations,
            timestep=simulation_options.timestep,
            fps=1./(skips*simulation_options.timestep),
            pitch=kwargs.pop('pitch', simulation_options.video_pitch),
            yaw=kwargs.pop('yaw', simulation_options.video_yaw),
            skips=skips,
            motion_filter=2*skips*simulation_options.timestep,
            distance=simulation_options.video_distance
        )

    def init_debug(self, animat_options):
        """Initialise debug"""
        # User parameters
        self.user_params = UserParameters(animat_options)

        # # Debug info
        # test_debug_info()


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
    """Play/pause parameter"""

    def __init__(self):
        super(ParameterPlay, self).__init__('Play', 1, 0, 1)
        self.value = True

    def update(self):
        """Update"""
        self.value = self.get_value() > 0.5


# class ParameterGait(DebugParameter):
#     """Gait control"""

#     def __init__(self, gait):
#         value = 0 if gait == 'standing' else 2 if gait == 'swimming' else 1
#         super(ParameterGait, self).__init__('Gait', value, 0, 2)
#         self.value = gait
#         self.changed = False

#     def update(self):
#         """Update"""
#         previous_value = self.value
#         value = self.get_value()
#         self.value = (
#             'standing'
#             if value < 0.5
#             else 'walking'
#             if 0.5 < value < 1.5
#             else 'swimming'
#         )
#         self.changed = (self.value != previous_value)
#         if self.changed:
#             pylog.debug('Gait changed ({} > {})'.format(
#                 previous_value,
#                 self.value
#             ))


class UserParameters(dict):
    """Parameters control"""

    def __init__(self, options):
        super(UserParameters, self).__init__()
        lim = np.pi/8
        self['play'] = ParameterPlay()
        self['rtl'] = DebugParameter('Real-time limiter', 1, 1e-3, 3)
        self['zoom'] = DebugParameter('Zoom', 1, 0, 1)
        # self['gait'] = ParameterGait(gait)
        # self['frequency'] = DebugParameter('Frequency', frequency, 0, 5)
        self['body_offset'] = DebugParameter('Body offset', 0, -lim, lim)
        self['drive_speed'] = DebugParameter(
            'Drive speed',
            options.control.drives.forward,
            0.9, 5.1
        )
        self['drive_turn'] = DebugParameter(
            'Drive turn',
            options.control.drives.turning,
            -0.2, 0.2
        )

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

    # @property
    # def gait(self):
    #     """Gait"""
    #     return self['gait']

    # @property
    # def frequency(self):
    #     """Frequency"""
    #     return self['frequency']

    def body_offset(self):
        """Body offset"""
        return self['body_offset']

    def drive_speed(self):
        """Drive speed"""
        return self['drive_speed']

    def drive_turn(self):
        """Drive turn"""
        return self['drive_turn']
