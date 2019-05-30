"""Cython sensors"""

from ..animats.array cimport NetworkArray2D, NetworkArray3D


cdef class Sensors(dict):
    """Sensors"""
    pass


# cdef class ContactSensor(NetworkArray2D):
#     """Model sensors"""

#     unsigned int animat_id = animat_id
#     unsigned int animat_link = animat_link
#     unsigned int target = target


# class JointsStatesSensor(NetworkArray3D):
#     """Joint state sensor"""

#     def __init__(self, n_iterations, model_id, joints, enable_ft=False):
#         super(JointsStatesSensor, self).__init__(
#             np.zeros([n_iterations, len(joints), 9])
#         )
#         self._model_id = model_id
#         self._joints = joints
#         self._enable_ft = enable_ft
#         if self._enable_ft:
#             for joint in self._joints:
#                 pybullet.enableJointForceTorqueSensor(
#                     self._model_id,
#                     joint
#                 )

#     def update(self, iteration):
#         """Update sensor"""
#         self.array[iteration] = np.array([
#             (state[0], state[1]) + state[2] + (state[3],)
#             for joint_i, state in enumerate(
#                 pybullet.getJointStates(self._model_id, self._joints)
#             )
#         ])


# class LinkStateSensor(NetworkArray2D):
#     """Links states sensor"""

#     def __init__(self, n_iterations, model_id, link):
#         super(LinkStateSensor, self).__init__(np.zeros([n_iterations, 13]))
#         self._model_id = model_id
#         self._link = link

#     def update(self, iteration):
#         """Update sensor"""
#         self.array[iteration] = np.concatenate(
#             pybullet.getLinkState(
#                 bodyUniqueId=self._model_id,
#                 linkIndex=self._link,
#                 computeLinkVelocity=1,
#                 computeForwardKinematics=1
#             )[4:]
#         )


# class Sensors(dict):
#     """Sensors"""

#     def add(self, new_dict):
#         """Add sensors"""
#         dict.update(self, new_dict)

#     def update(self, iteration):
#         """Update all sensors"""
#         for sensor in self.values():
#             sensor.update(iteration)
