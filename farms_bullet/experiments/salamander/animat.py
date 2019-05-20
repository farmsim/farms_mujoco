"""Salamander"""

import os
import time
import numpy as np

import pybullet

from ...animats.animat import Animat
from ...animats.link import AnimatLink
from ...animats.model import Model
from ...plugins.swimming import viscous_swimming
from ...sensors.sensor import (
    Sensors,
    JointsStatesSensor,
    ContactSensor,
    LinkStateSensor
)
from .control import SalamanderController


class SalamanderModel(Model):
    """Salamander model"""

    def __init__(
            self, identity, base_link,
            iterations, timestep,
            gait="walking", **kwargs
    ):
        super(SalamanderModel, self).__init__(
            identity=identity,
            base_link=base_link
        )
        # Correct names
        self.links = {}
        for i in range(11):
            self.links['link_body_{}'.format(i+1)] = i
            self.joints['joint_link_body_{}'.format(i+1)] = (
                self.joints.pop('joint{}'.format(i+1))
            )
        n_dof_legs = 4
        for leg_i in range(2):
            for side_i in range(2):
                for part_i in range(n_dof_legs):
                    link_index = (
                        12 + 2*n_dof_legs*leg_i + n_dof_legs*side_i + part_i
                    )
                    side = "R" if side_i else "L"
                    self.links[
                        'link_leg_{}_{}_{}'.format(
                            leg_i,
                            side,
                            part_i
                        )
                    ] = link_index-1
                    self.joints[
                        'joint_link_leg_{}_{}_{}'.format(
                            leg_i,
                            side,
                            part_i
                        )
                    ] = self.joints.pop('joint{}'.format(link_index))
        # Model dynamics
        self.apply_motor_damping()
        # Controller
        self.controller = SalamanderController.from_gait(
            self.identity,
            self.joints,
            gait=gait,
            iterations=iterations,
            timestep=timestep,
            **kwargs
        )
        self.feet = [
            "link_leg_0_L_3",
            "link_leg_0_R_3",
            "link_leg_1_L_3",
            "link_leg_1_R_3"
        ]
        self.sensors = NotImplemented


    @classmethod
    def spawn(cls, iterations, timestep, gait="walking", **kwargs):
        """Spawn salamander"""
        # Body
        meshes_directory = (
            "{}/meshes".format(
                os.path.dirname(os.path.realpath(__file__))
            )
        )
        body_link_positions = np.diff(
            [  # From SDF
                [0, 0, 0],
                [0.200000003, 0, 0.0069946074],
                [0.2700000107, 0, 0.010382493],
                [0.3400000036, 0, 0.0106022889],
                [0.4099999964, 0, 0.010412137],
                [0.4799999893, 0, 0.0086611426],
                [0.5500000119, 0, 0.0043904358],
                [0.6200000048, 0, 0.0006898994],
                [0.6899999976, 0, 8.0787e-06],
                [0.7599999905, 0, -4.89001e-05],
                [0.8299999833, 0, 0.0001386079],
                [0.8999999762, 0, 0.0003494423]
            ],
            axis=0,
            prepend=0
        )
        body_color = [0, 0.3, 0, 1]
        base_link = AnimatLink(
            geometry=pybullet.GEOM_MESH,
            filename="{}/salamander_body_0.obj".format(meshes_directory),
            position=body_link_positions[0],
            joint_axis=[0, 0, 1],
            color=body_color
        )
        links_body = [
            AnimatLink(
                geometry=pybullet.GEOM_MESH,
                filename="{}/salamander_body_{}.obj".format(
                    meshes_directory,
                    i+1
                ),
                position=body_link_positions[i+1],
                parent=i,
                joint_axis=[0, 0, 1],
                color=body_color
            )
            for i in range(11)
        ]
        links_legs = [None for i in range(4) for j in range(4)]
        leg_length = 0.03
        leg_radius = 0.015
        n_body = 12
        n_dof_legs = 4
        for leg_i in range(2):
            for side in range(2):
                offset = 2*n_dof_legs*leg_i + n_dof_legs*side
                sign = 1 if side else -1
                leg_offset = sign*leg_length
                position = np.zeros(3)
                position[1] = leg_offset
                # Shoulder1
                links_legs[offset+0] = AnimatLink(
                    geometry=pybullet.GEOM_SPHERE,
                    radius=1.2*leg_radius,
                    position=position,
                    parent=5 if leg_i else 1,
                    joint_axis=[0, 0, sign],
                    mass=0,
                    color=[0.9, 0.0, 0.0, 0.3]
                )
                # Shoulder2
                links_legs[offset+1] = AnimatLink(
                    geometry=pybullet.GEOM_SPHERE,
                    radius=1.5*leg_radius,
                    parent=n_body+offset,
                    joint_axis=[-sign, 0, 0],
                    mass=0,
                    color=[0.9, 0.9, 0.9, 0.3]
                )
                # Upper leg
                links_legs[offset+2] = AnimatLink(
                    geometry=pybullet.GEOM_CAPSULE,
                    radius=leg_radius,
                    height=0.9*2*leg_length,
                    frame_position=position,
                    frame_orientation=[np.pi/2, 0, 0],
                    parent=n_body+offset+1,
                    joint_axis=[0, 1, 0]
                )
                # Lower leg
                links_legs[offset+3] = AnimatLink(
                    geometry=pybullet.GEOM_CAPSULE,
                    radius=leg_radius,
                    height=0.9*2*leg_length,
                    position=2*position,
                    frame_position=position,
                    frame_orientation=[np.pi/2, 0, 0],
                    parent=n_body+offset+2,
                    joint_axis=[-sign, 0, 0]
                )
        links = links_body + links_legs
        identity = pybullet.createMultiBody(
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
        return cls(
            identity=identity,
            base_link="link_body_0",
            iterations=iterations,
            timestep=timestep,
            gait=gait,
            **kwargs
        )

    @classmethod
    def spawn_sdf(cls, iterations, timestep, gait="walking", **kwargs):
        """Spawn salamander"""
        return cls.from_sdf(
            "{}/.farms/models/biorob_salamander/model.sdf".format(
                os.environ['HOME']
            ),
            base_link="link_body_0",
            iterations=iterations,
            timestep=timestep,
            gait=gait,
            **kwargs
        )

    def leg_collisions(self, plane, activate=True):
        """Activate/Deactivate leg collisions"""
        for leg_i in range(2):
            for side in ["L", "R"]:
                for joint_i in range(2):
                    link = "link_leg_{}_{}_{}".format(leg_i, side, joint_i)
                    pybullet.setCollisionFilterPair(
                        bodyUniqueIdA=self.identity,
                        bodyUniqueIdB=plane,
                        linkIndexA=self.links[link],
                        linkIndexB=-1,
                        enableCollision=activate
                    )

    def apply_motor_damping(self, linear=0, angular=0):
        """Apply motor damping"""
        for j in range(pybullet.getNumJoints(self.identity)):
            pybullet.changeDynamics(
                bodyUniqueId=self.identity,
                linkIndex=j,
                linearDamping=linear,
                angularDamping=angular
            )


class Salamander(Animat):
    """Salamander animat"""

    def __init__(self, options, timestep, n_iterations):
        super(Salamander, self).__init__(options)
        self.model = NotImplemented
        self.timestep = timestep
        self.sensors = NotImplemented
        self.n_iterations = n_iterations

    def spawn(self):
        """Spawn"""
        self.model = SalamanderModel.spawn(
            self.n_iterations,
            self.timestep,
            **self.options
        )
        self._identity = self.model.identity

    def add_sensors(self, arena_identity):
        """Add sensors"""
        # Sensors
        self.sensors = Sensors()
        # Contacts
        self.sensors.add({
            "contact_{}".format(i): ContactSensor(
                self.n_iterations,
                self._identity, self.links[foot],
                arena_identity, -1
            )
            for i, foot in enumerate(self.model.feet)
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

    @property
    def links(self):
        """Links"""
        return self.model.links

    @property
    def joints(self):
        """Joints"""
        return self.model.joints

    def animat_control(self):
        """Control animat"""
        # Control
        tic_control = time.time()
        self.model.controller.control()
        time_control = time.time() - tic_control
        return time_control

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
