"""Animat"""

import numpy as np
import pybullet
import farms_pylog as pylog
from .model import SimulationModel


def joint_type_str(joint_type):
    """Return joint type as str"""
    return (
        'Revolute' if joint_type == pybullet.JOINT_REVOLUTE
        else 'Prismatic' if joint_type == pybullet.JOINT_PRISMATIC
        else 'Spherical' if joint_type == pybullet.JOINT_SPHERICAL
        else 'Planar' if joint_type == pybullet.JOINT_PLANAR
        else 'Fixed' if joint_type == pybullet.JOINT_FIXED
        else 'Unknown'
    )


class Animat(SimulationModel):
    """Animat"""

    def __init__(self, identity=None, options=None):
        super(Animat, self).__init__(identity=identity)
        self.options = options
        self._links = {}
        self._joints = {}
        self.sensors = {}
        self.data = None

    def n_joints(self):
        """Get number of joints"""
        return pybullet.getNumJoints(self._identity)

    def links_identities(self):
        """Joints"""
        return np.arange(-1, pybullet.getNumJoints(self._identity), dtype=int)

    def joints_identities(self):
        """Joints"""
        return np.arange(pybullet.getNumJoints(self._identity), dtype=int)

    @staticmethod
    def get_parent_links_info(identity, base_link='base_link'):
        """Get links (parent of joint)"""
        links = {base_link: -1}
        links.update({
            info[12].decode('UTF-8'): info[16] + 1
            for info in [
                pybullet.getJointInfo(identity, j)
                for j in range(pybullet.getNumJoints(identity))
            ]
        })
        return links

    @staticmethod
    def get_joints_info(identity):
        """Get joints"""
        joints = {
            info[1].decode('UTF-8'): info[0]
            for info in [
                pybullet.getJointInfo(identity, j)
                for j in range(pybullet.getNumJoints(identity))
            ]
        }
        return joints

    def print_information(self):
        """Print information"""
        pylog.debug('Links ids:\n{}'.format(
            '\n'.join([
                '  {}: {}'.format(name, identity)
                for name, identity in self._links.items()
            ])
        ))
        pylog.debug('Joints ids:\n{}'.format(
            '\n'.join([
                '  {}: {} (type: {})'.format(
                    name,
                    identity,
                    joint_type_str(
                        pybullet.getJointInfo(
                            self.identity(),
                            identity
                        )[2]
                    )
                )
                for name, identity in self._joints.items()
            ])
        ))

    def print_dynamics_info(self, links=None):
        """Print dynamics info"""
        links = links if links is not None else self._links
        pylog.debug('Dynamics:')
        for link in links:
            dynamics_msg = (
                '\n      mass: {}'
                '\n      lateral_friction: {}'
                '\n      local inertia diagonal: {}'
                '\n      local inertial pos: {}'
                '\n      local inertial orn: {}'
                '\n      restitution: {}'
                '\n      rolling friction: {}'
                '\n      spinning friction: {}'
                '\n      contact damping: {}'
                '\n      contact stiffness: {}'
            )

            pylog.debug('  - {}:{}'.format(
                link,
                dynamics_msg.format(*pybullet.getDynamicsInfo(
                    self.identity(),
                    self._links[link]
                ))
            ))
        pylog.debug('Model mass: {} [kg]'.format(self.mass()))

    def total_mass(self):
        """Print dynamics"""
        return np.sum([
            pybullet.getDynamicsInfo(self.identity(), self._links[link])[0]
            for link in self._links
        ])

    def get_position(self, link):
        """Get position"""
        return pybullet.getLinkState(self.identity(), link)[0]

    def set_collisions(self, links, group=0, mask=0):
        """Activate/Deactivate leg collisions"""
        for link in links:
            pybullet.setCollisionFilterGroupMask(
                bodyUniqueId=self._identity,
                linkIndexA=self._links[link],
                collisionFilterGroup=group,
                collisionFilterMask=mask
            )

    def set_links_dynamics(self, links, **kwargs):
        """Apply motor damping"""
        for link in links:
            for key, value in kwargs.items():
                pybullet.changeDynamics(
                    bodyUniqueId=self.identity(),
                    linkIndex=self._links[link],
                    **{key: value}
                )
