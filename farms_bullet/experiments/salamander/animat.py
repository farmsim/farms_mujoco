"""Salamander"""

import os

import numpy as np
import pybullet

from ...animats.animat import Animat
from ...animats.link import AnimatLink
from ...plugins.swimming import viscous_forces, swimming_motion, swimming_debug
from ...sensors.sensors import (
    Sensors,
    JointsStatesSensor,
    ContactsSensors
)
from .convention import (
    leglink2index,
    leglink2name,
    legjoint2index,
    legjoint2name
)
from .animat_data import (
    SalamanderOscillatorNetworkState,
    SalamanderData
)
from .control import SalamanderController
from .sensors import SalamanderGPS


class Salamander(Animat):
    """Salamander animat"""

    def __init__(self, options, timestep, iterations, units):
        super(Salamander, self).__init__(options=options)
        self.timestep = timestep
        self.n_iterations = iterations
        self.feet_names = [
            "link_leg_0_L_3",
            "link_leg_0_R_3",
            "link_leg_1_L_3",
            "link_leg_1_R_3"
        ]
        self.data = SalamanderData.from_options(
            SalamanderOscillatorNetworkState.default_state(iterations),
            options,
            iterations
        )
        # Hydrodynamic forces
        self.hydrodynamics = None
        # Physics
        self.units = units
        self.scale = options.morphology.scale

    def spawn(self):
        """Spawn salamander"""
        self.spawn_body()
        self.setup_controller()
        self.add_sensors()
        self.set_body_properties()
        self.hydrodynamics = [
            pybullet.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[0, 0, 0],
                lineColorRGB=[0, 0, 0],
                lineWidth=3*self.units.meters,
                lifeTime=0,
                parentObjectUniqueId=self.identity,
                parentLinkIndex=i
            )
            for i in range(12)
        ]

    def spawn_body(self):
        """Spawn body"""
        meshes_directory = (
            "{}/meshes".format(
                os.path.dirname(os.path.realpath(__file__))
            )
        )
        body_link_positions = self.scale*np.diff(
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
            self.units,
            geometry=pybullet.GEOM_MESH,
            filename="{}/salamander_body_0.obj".format(meshes_directory),
            position=body_link_positions[0],
            joint_axis=[0, 0, 1],
            color=body_color,
            scale=[self.scale, self.scale, self.scale]
        )
        links = [
            AnimatLink(
                self.units,
                geometry=pybullet.GEOM_MESH,
                filename="{}/salamander_body_{}.obj".format(
                    meshes_directory,
                    i+1
                ),
                position=body_link_positions[i+1],
                parent=i,
                joint_axis=[0, 0, 1],
                color=body_color,
                scale=[self.scale, self.scale, self.scale]
            )
            for i in range(11)
        ] + [None for i in range(4) for j in range(4)]
        leg_offset = self.scale*0.03
        leg_length = self.scale*0.06
        leg_radius = self.scale*0.015
        for leg_i in [1, 0]:
            for side in range(2):
                sign = 1 if side else -1
                offset = np.zeros(3)
                offset[1] = sign*leg_offset
                position = np.zeros(3)
                position[1] = 0.5*sign*leg_length
                # Shoulder1
                links[leglink2index(leg_i, side, 0)] = AnimatLink(
                    self.units,
                    geometry=pybullet.GEOM_SPHERE,
                    radius=1.2*leg_radius,
                    position=offset,
                    parent=5 if leg_i else 1,  # Inverse seems to change nothing
                    joint_axis=[0, 0, sign],
                    mass=0,
                    color=[0.9, 0.0, 0.0, 0.3]
                )
                # Shoulder2
                links[leglink2index(leg_i, side, 1)] = AnimatLink(
                    self.units,
                    geometry=pybullet.GEOM_SPHERE,
                    radius=1.5*leg_radius,
                    parent=leglink2index(leg_i, side, 0)+1,
                    joint_axis=[-sign, 0, 0],
                    mass=0,
                    color=[0.9, 0.9, 0.9, 0.3]
                )
                # Upper leg
                links[leglink2index(leg_i, side, 2)] = AnimatLink(
                    self.units,
                    geometry=pybullet.GEOM_CAPSULE,
                    radius=leg_radius,
                    height=leg_length,
                    frame_position=position,
                    frame_orientation=[np.pi/2, 0, 0],
                    parent=leglink2index(leg_i, side, 1)+1,
                    joint_axis=[0, 1, 0]
                )
                # Lower leg
                links[leglink2index(leg_i, side, 3)] = AnimatLink(
                    self.units,
                    geometry=pybullet.GEOM_CAPSULE,
                    radius=leg_radius,
                    height=leg_length,
                    position=2*position,
                    frame_position=position,
                    frame_orientation=[np.pi/2, 0, 0],
                    parent=leglink2index(leg_i, side, 2)+1,
                    joint_axis=[-sign, 0, 0],
                    # color=[
                    #     [[1, 0, 0, 1], [0, 1, 0, 1]],
                    #     [[0, 0, 1, 1], [0, 0, 0, 1]]
                    # ][leg_i][side]
                )
        self._identity = pybullet.createMultiBody(
            baseMass=base_link.mass*self.units.kilograms,
            baseCollisionShapeIndex=base_link.collision,
            baseVisualShapeIndex=base_link.visual,
            basePosition=(
                np.array(self.options.spawn.position)*self.units.meters
            ),
            baseOrientation=pybullet.getQuaternionFromEuler(
                self.options.spawn.orientation
            ),
            baseInertialFramePosition=np.array(
                base_link.inertial_position
            )*self.units.meters,
            baseInertialFrameOrientation=base_link.inertial_orientation,
            linkMasses=[link.mass*self.units.kilograms for link in links],
            linkCollisionShapeIndices=[link.collision for link in links],
            linkVisualShapeIndices=[link.visual for link in links],
            linkPositions=np.array(
                [link.position for link in links]
            )*self.units.meters,
            linkOrientations=[link.orientation for link in links],
            linkInertialFramePositions=np.array([
                link.inertial_position
                for link in links
            ])*self.units.meters,
            linkInertialFrameOrientations=[
                link.inertial_orientation
                for link in links
            ],
            linkParentIndices=[link.parent for link in links],
            linkJointTypes=[link.joint_type for link in links],
            linkJointAxis=[link.joint_axis for link in links]
        )
        # Set names
        self.links['link_body_{}'.format(0)] = -1
        for i in range(11):
            self.links['link_body_{}'.format(i+1)] = i
            self.joints['joint_link_body_{}'.format(i)] = i
        for leg_i in range(2):
            for side_i in range(2):
                for part_i in range(self.options.morphology.n_dof_legs):
                    self.links[
                        # TODO: Find out why legs indices are reversed
                        leglink2name((leg_i + 1)%2, side_i, part_i)
                    ] = (
                        leglink2index(leg_i, side_i, part_i)
                    )
                    self.joints[
                        # TODO: Find out why legs indices are reversed
                        legjoint2name((leg_i + 1)%2, side_i, part_i)
                    ] = (
                        legjoint2index(leg_i, side_i, part_i)
                    )
        self.print_information()

    # @classmethod
    # def spawn_sdf(cls, iterations, timestep, gait="walking", **kwargs):
    #     """Spawn salamander"""
    #     return cls.from_sdf(
    #         "{}/.farms/models/biorob_salamander/model.sdf".format(
    #             os.environ['HOME']
    #         ),
    #         base_link="link_body_0",
    #         iterations=iterations,
    #         timestep=timestep,
    #         gait=gait,
    #         **kwargs
    #     )

    def add_sensors(self):
        """Add sensors"""
        # Sensors
        self.sensors = Sensors()
        # Contacts
        self.sensors.add({
            "contacts": ContactsSensors(
                self.data.sensors.contacts.array,
                [self._identity for _ in self.feet_names],
                [self.links[foot] for foot in self.feet_names],
                self.units.newtons
            )
        })
        # Joints
        self.sensors.add({
            "joints": JointsStatesSensor(
                self.data.sensors.proprioception.array,
                self._identity,
                np.arange(self.n_joints()),
                self.units,
                enable_ft=True
            )
        })
        # Base link
        links = [
            [
                "link_body_{}".format(i),
                i,
                self.links["link_body_{}".format(i)]
            ]
            for i in range(12)
        ] + [
            [
                "link_leg_{}_{}_{}".format(leg_i, side, joint_i),
                12 + leg_i*2*4 + side_i*4 + joint_i,
                self.links["link_leg_{}_{}_{}".format(
                    leg_i,
                    side,
                    joint_i
                )]
            ]
            for leg_i in range(2)
            for side_i, side in enumerate(["L", "R"])
            for joint_i in range(4)
        ]
        self.sensors.add({
            "links": SalamanderGPS(
                array=self.data.sensors.gps.array,
                animat_id=self.identity,
                links=links,
                options=self.options,
                units=self.units
            )
        })

    def set_body_properties(self):
        """Set body properties"""
        # Deactivate collisions
        links_no_collisions = [
            "link_body_{}".format(body_i)
            for body_i in range(0)
        ] + [
            "link_leg_{}_{}_{}".format(leg_i, side, joint_i)
            for leg_i in range(2)
            for side in ["L", "R"]
            for joint_i in range(3)
        ]
        self.set_collisions(links_no_collisions, group=0, mask=0)
        # Deactivate damping
        links_no_damping = [
            "link_body_{}".format(body_i)
            for body_i in range(12)
        ] + [
            "link_leg_{}_{}_{}".format(leg_i, side, joint_i)
            for leg_i in range(2)
            for side in ["L", "R"]
            for joint_i in range(4)
        ]
        self.set_links_dynamics(
            links_no_damping,
            linearDamping=0,
            angularDamping=0,
            jointDamping=0
        )
        # Friction
        self.set_links_dynamics(
            self.links,
            lateralFriction=0,
            spinningFriction=0,
            rollingFriction=0,
        )
        self.set_links_dynamics(
            self.feet_names,
            lateralFriction=1,
            spinningFriction=0,
            rollingFriction=0,
        )

    def setup_controller(self):
        """Setup controller"""
        self.controller = SalamanderController.from_data(
            self.identity,
            animat_data=self.data,
            timestep=self.timestep,
            units=self.units
        )

    def viscous_swimming_forces(self, iteration):
        """Animat swimming physics"""
        viscous_forces(
            iteration,
            self.data.sensors.gps,
            self.data.sensors.hydrodynamics.array,
            [
                [i, self.links["link_body_{}".format(i)]]
                for i in range(12)
            ],
            coefficients=[
                self.options.morphology.scale**3*np.array([-1e-1, -1e0, -1e0]),
                self.options.morphology.scale**6*np.array([-1e-2, -1e-2, -1e-2])
            ]
        )

    def apply_swimming_forces(self, iteration, link_frame=True, debug=False):
        """Animat swimming physics"""
        swimming_motion(
            iteration,
            self.data.sensors.hydrodynamics.array,
            self.identity,
            [
                [i, self.links["link_body_{}".format(i)]]
                for i in range(12)
            ],
            link_frame=link_frame,
            units=self.units
        )
        if debug:
            swimming_debug(
                iteration,
                self.data.sensors.gps,
                [
                    [i, self.links["link_body_{}".format(i)]]
                    for i in range(12)
                ]
            )

    def draw_hydrodynamics(self, iteration):
        """Draw hydrodynamics forces"""
        for i, line in enumerate(self.hydrodynamics):
            force = self.data.sensors.hydrodynamics.array[iteration, i, :3]
            self.hydrodynamics[i] = pybullet.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=np.array(force),
                lineColorRGB=[0, 0, 1],
                lineWidth=7*self.units.meters,
                parentObjectUniqueId=self.identity,
                parentLinkIndex=i-1,
                replaceItemUniqueId=line
            )
