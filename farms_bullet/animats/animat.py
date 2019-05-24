"""Animat"""

import numpy as np
import pybullet
from ..simulations.element import SimulationElement


def joint_type_str(joint_type):
    """Return joint type as str"""
    return (
        "Revolute" if joint_type == pybullet.JOINT_REVOLUTE
        else "Prismatic" if joint_type == pybullet.JOINT_PRISMATIC
        else "Spherical" if joint_type == pybullet.JOINT_SPHERICAL
        else "Planar" if joint_type == pybullet.JOINT_PLANAR
        else "Fixed" if joint_type == pybullet.JOINT_FIXED
        else "Unknown"
    )


class Animat(SimulationElement):
    """Animat"""

    def __init__(self, identity=None, options=None):
        super(Animat, self).__init__(identity=identity)
        self.options = options
        self.links = {}
        self.joints = {}
        self.sensors = {}
        self.controller = None

    def n_joints(self):
        """Get number of joints"""
        return pybullet.getNumJoints(self._identity)

    @staticmethod
    def get_parent_links_info(identity, base_link="base_link"):
        """Get links (parent of joint)"""
        links = {base_link: -1}
        links.update({
            info[12].decode("UTF-8"): info[16] + 1
            for info in [
                pybullet.getJointInfo(identity, j)
                for j in range(pybullet.getNumJoints(identity))
            ]
        })
        return links

    @staticmethod
    def get_joints_info(identity, base_link="base_link"):
        """Get joints"""
        joints = {
            info[1].decode("UTF-8"): info[0]
            for info in [
                pybullet.getJointInfo(identity, j)
                for j in range(pybullet.getNumJoints(identity))
            ]
        }
        return joints

    def print_information(self):
        """Print information"""
        print("Links ids:\n{}".format(
            "\n".join([
                "  {}: {}".format(
                    name,
                    self.links[name]
                )
                for name in self.links
            ])
        ))
        print("Joints ids:\n{}".format(
            "\n".join([
                "  {}: {} (type: {})".format(
                    name,
                    self.joints[name],
                    joint_type_str(
                        pybullet.getJointInfo(
                            self.identity,
                            self.joints[name]
                        )[2]
                    )
                )
                for name in self.joints
            ])
        ))

    def print_dynamics_info(self, links=None):
        """Print dynamics info"""
        links = links if links is not None else self.links
        print("Dynamics:")
        for link in links:
            dynamics_msg = (
                "\n      mass: {}"
                "\n      lateral_friction: {}"
                "\n      local inertia diagonal: {}"
                "\n      local inertial pos: {}"
                "\n      local inertial orn: {}"
                "\n      restitution: {}"
                "\n      rolling friction: {}"
                "\n      spinning friction: {}"
                "\n      contact damping: {}"
                "\n      contact stiffness: {}"
            )

            print("  - {}:{}".format(
                link,
                dynamics_msg.format(*pybullet.getDynamicsInfo(
                    self.identity,
                    self.links[link]
                ))
            ))
        print("Model mass: {} [kg]".format(self.mass()))

    def mass(self):
        """Print dynamics"""
        return np.sum([
            pybullet.getDynamicsInfo(self.identity, self.links[link])[0]
            for link in self.links
        ])

    def get_position(self):
        """Get position"""
        return pybullet.getLinkState(self.identity, 0)[0]

    def set_collisions(self, links, group=0, mask=0):
        """Activate/Deactivate leg collisions"""
        for link in links:
            pybullet.setCollisionFilterGroupMask(
                bodyUniqueId=self._identity,
                linkIndexA=self.links[link],
                collisionFilterGroup=group,
                collisionFilterMask=mask
            )

    def set_links_dynamics(self, links, **kwargs):
        """Apply motor damping"""
        for link in links:
            pybullet.changeDynamics(
                bodyUniqueId=self.identity,
                linkIndex=self.links[link],
                **kwargs
            )
