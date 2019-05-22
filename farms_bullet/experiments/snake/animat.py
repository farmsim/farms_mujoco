"""Snake"""

import os
import time
import numpy as np

import pybullet

from ...animats.animat import Animat
from ...animats.link import AnimatLink
from ...plugins.swimming import viscous_swimming
from ...sensors.sensor import (
    Sensors,
    JointsStatesSensor,
    ContactSensor,
    LinkStateSensor
)


class Snake(Animat):
    """Snake animat"""

    def __init__(self, options, timestep, iterations):
        super(Snake, self).__init__(options=options)
        self.timestep = timestep
        self.n_iterations = iterations

    def spawn(self):
        """Spawn snake"""
        self.spawn_body()
        self.add_sensors()
        # Deactivate collisions
        links_no_collisions = [
            "link_body_{}".format(body_i+1)
            for body_i in range(0)
        ]
        self.set_collisions(links_no_collisions, group=0, mask=0)
        # Deactivate damping
        joints_no_damping = [
            "joint_link_body_{}".format(body_i+1)
            for body_i in range(0)
        ]
        self.set_joint_damping(joints_no_damping, linear=0, angular=0)

    def spawn_body(self):
        """Spawn body"""
        # body_color = [0, 0.3, 0, 1]
        body_link_positions = np.zeros([12, 3])
        body_link_positions[1:, 0] = 0.06
        body_shape = {
            "geometry": pybullet.GEOM_BOX,
            "size": [0.03, 0.02, 0.02]
        }
        body_shape = {
            "geometry": pybullet.GEOM_CAPSULE,
            "radius": 0.03,
            "height": 0.06,
            "frame_orientation": [0, 0.5*np.pi, 0]
        }
        base_link = AnimatLink(
            **body_shape,
            position=body_link_positions[0],
            joint_axis=[0, 0, 1],
            # color=body_color
        )
        links = [
            AnimatLink(
                **body_shape,
                position=body_link_positions[i+1],
                parent=i,
                joint_axis=[0, 0, 1],
                # color=body_color
            )
            for i in range(11)
        ]
        self._identity = pybullet.createMultiBody(
            baseMass=base_link.mass,
            baseCollisionShapeIndex=base_link.collision,
            baseVisualShapeIndex=base_link.visual,
            basePosition=[0, 0, 0],
            baseOrientation=pybullet.getQuaternionFromEuler([0, 0, 0]),
            linkMasses=[link.mass for link in links],
            linkCollisionShapeIndices=[link.collision for link in links],
            linkVisualShapeIndices=[link.visual for link in links],
            linkPositions=[link.position for link in links],
            linkOrientations=[link.orientation for link in links],
            linkInertialFramePositions=[link.f_position for link in links],
            linkInertialFrameOrientations=[link.f_orientation for link in links],
            linkParentIndices=[link.parent for link in links],
            linkJointTypes=[link.joint_type for link in links],
            linkJointAxis=[link.joint_axis for link in links]
        )
        # Get links and joints
        # Correct names
        self.links['link_body_{}'.format(0)] = -1
        for i in range(11):
            self.links['link_body_{}'.format(i+1)] = i
            self.joints['joint_link_body_{}'.format(i)] = i
        self.print_information()

    def add_sensors(self):
        """Add sensors"""
        # Sensors
        self.sensors = Sensors()
        # Contacts
        self.sensors.add({
            "contact_{}".format(link): ContactSensor(
                self.n_iterations,
                self._identity, self.links[link]
            )
            for link in self.links
        })
        # Joints
        self.sensors.add({
            "joints": JointsStatesSensor(
                self.n_iterations,
                self._identity,
                np.arange(self.n_joints()),
                enable_ft=True
            )
        })
        # Base link
        self.sensors.add({
            "base_link": LinkStateSensor(
                self.n_iterations,
                self._identity,
                0,  # Base link
            )
        })

    def animat_physics(self):
        """Animat physics"""
        # Swimming
        forces = None
        if self.options.gait == "swimming":
            forces = viscous_swimming(
                self.identity,
                self.links
            )
        return forces
