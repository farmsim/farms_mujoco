"""Model"""

import numpy as np
import pybullet


class Model:
    """Simulation model"""

    def __init__(self, identity, base_link="base_link"):
        super(Model, self).__init__()
        self.identity = identity
        self.links, self.joints, self.n_joints = self.get_joints(
            self.identity,
            base_link
        )
        self.print_information()

    @classmethod
    def from_sdf(cls, sdf, base_link="base_link", **kwargs):
        """Model from SDF"""
        identity = pybullet.loadSDF(sdf)[0]
        return cls(identity, base_link=base_link, **kwargs)

    @classmethod
    def from_urdf(cls, urdf, base_link="base_link", **kwargs):
        """Model from SDF"""
        identity = pybullet.loadURDF(urdf, **kwargs)
        return cls(identity, base_link=base_link)

    @staticmethod
    def get_joints(identity, base_link="base_link"):
        """Get joints"""
        print("Identity: {}".format(identity))
        n_joints = pybullet.getNumJoints(identity)
        print("Number of joints: {}".format(n_joints))

        # Links
        # Base link
        links = {base_link: -1}
        links.update({
            info[12].decode("UTF-8"): info[16] + 1
            for info in [
                pybullet.getJointInfo(identity, j)
                for j in range(n_joints)
            ]
        })
        # Joints
        joints = {
            info[1].decode("UTF-8"): info[0]
            for info in [
                pybullet.getJointInfo(identity, j)
                for j in range(n_joints)
            ]
        }
        return links, joints, n_joints

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
                "  {}: {}".format(
                    name,
                    self.joints[name]
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
