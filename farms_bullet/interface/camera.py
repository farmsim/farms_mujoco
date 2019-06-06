"""Camera"""

import numpy as np

import pybullet


class Camera:
    """Camera"""

    def __init__(self, timestep, target_identity=None, **kwargs):
        super(Camera, self).__init__()
        self.target = target_identity
        cam_info = self.get_camera()
        self.timestep = timestep
        self.motion_filter = kwargs.pop("motion_filter", 2*timestep)
        self.yaw = kwargs.pop("yaw", cam_info[8])
        self.yaw_speed = kwargs.pop("yaw_speed", 0)
        self.pitch = kwargs.pop("pitch", cam_info[9])
        self.distance = kwargs.pop("distance", cam_info[10])

    @staticmethod
    def get_camera():
        """Get camera information"""
        return pybullet.getDebugVisualizerCamera()

    def update_yaw(self):
        """Update yaw"""
        self.yaw += self.yaw_speed*self.timestep


class CameraTarget(Camera):
    """Camera with target following"""

    def __init__(self, target_identity, **kwargs):
        super(CameraTarget, self).__init__(**kwargs)
        self.target = target_identity
        self.target_pos = kwargs.pop(
            "target_pos",
            np.array(pybullet.getBasePositionAndOrientation(self.target)[0])
            if self.target is not None
            else np.array(self.get_camera()[11])
        )

    def update_target_pos(self):
        """Update target position"""
        self.target_pos = (
            (1-self.motion_filter)*self.target_pos
            + self.motion_filter*np.array(
                pybullet.getBasePositionAndOrientation(self.target)[0]
            )
        )


class UserCamera(CameraTarget):
    """UserCamera"""

    def __init__(self, target_identity, **kwargs):
        super(UserCamera, self).__init__(target_identity, **kwargs)
        self.update(use_camera=False)

    def update(self, use_camera=True):
        """Camera view"""
        if use_camera:
            self.yaw, self.pitch, self.distance = self.get_camera()[8:11]
        self.update_yaw()
        if self.target is not None:
            self.update_target_pos()
        pybullet.resetDebugVisualizerCamera(
            cameraDistance=self.distance,
            cameraYaw=self.yaw,
            cameraPitch=self.pitch,
            cameraTargetPosition=self.target_pos
        )


class CameraRecord(CameraTarget):
    """Camera recording"""

    def __init__(self, target_identity, size, fps, **kwargs):
        super(CameraRecord, self).__init__(target_identity, **kwargs)
        self.width = kwargs.pop("width", 1280)
        self.height = kwargs.pop("height", 720)
        self.fps = fps
        self.skips = kwargs.pop("skips", 1)
        self.data = np.zeros(
            [size, self.height, self.width, 4],
            dtype=np.uint8
        )
        self.iteration = 0

    def record(self, step):
        """Record camera"""
        if not step % self.skips:
            sample = step//self.skips-1
            self.update_yaw()
            self.update_target_pos()
            self.data[sample, :, :] = pybullet.getCameraImage(
                width=self.width,
                height=self.height,
                viewMatrix=pybullet.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=self.target_pos,
                    distance=self.distance,
                    yaw=self.yaw,
                    pitch=self.pitch,
                    roll=0,
                    upAxisIndex=2
                ),
                projectionMatrix = pybullet.computeProjectionMatrixFOV(
                    fov=60,
                    aspect=self.width/self.height,
                    nearVal=0.1,
                    farVal=5
                ),
                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
                flags=pybullet.ER_NO_SEGMENTATION_MASK
            )[2]
            self.iteration += 1

    def save(self, filename="video.avi"):
        """Save recording"""
        print("Recording video to {}".format(filename))
        import cv2
        writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*'MJPG'),
            self.fps,
            (self.width, self.height)
        )
        for image in self.data[:self.iteration]:
            writer.write(image)
