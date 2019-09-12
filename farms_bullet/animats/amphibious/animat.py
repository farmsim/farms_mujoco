"""Amphibious"""

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

from ..farms_sdf import ModelSDF, Link, Joint

from .convention import AmphibiousConvention
from .animat_data import (
    AmphibiousOscillatorNetworkState,
    AmphibiousData
)
from .control import AmphibiousController
from .sensors import AmphibiousGPS


class Amphibious(Animat):
    """Amphibious animat"""

    def __init__(self, options, timestep, iterations, units):
        super(Amphibious, self).__init__(options=options)
        self.timestep = timestep
        self.n_iterations = iterations
        self.convention = AmphibiousConvention(self.options)
        self.feet_names = [
            self.convention.leglink2name(
                leg_i=leg_i,
                side_i=side_i,
                joint_i=3
            )
            for leg_i in range(options.morphology.n_legs//2)
            for side_i in range(2)
        ]
        self.joints_order = None
        self.data = AmphibiousData.from_options(
            AmphibiousOscillatorNetworkState.default_state(iterations, options),
            options,
            iterations
        )
        # Hydrodynamic forces
        self.masses = np.zeros(options.morphology.n_links())
        self.hydrodynamics = None
        # Sensors
        self.sensors = Sensors()
        # Physics
        self.units = units
        self.scale = options.morphology.scale

    def spawn(self):
        """Spawn amphibious"""
        self.spawn_body()
        # self.spawn_sdf()
        # Controller
        self.setup_controller()
        # Sensors
        self.add_sensors()
        # Body properties
        self.set_body_properties()
        # Debug
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
            for i in range(self.options.morphology.n_links_body())
        ]

    def spawn_sdf(self, verbose=False):
        """Spawn sdf"""
        links = [None for _ in range(self.options.morphology.n_links())]
        joints = [None for _ in range(self.options.morphology.n_joints())]
        if self.options.morphology.mesh_directory:
            body_link_positions = self.scale*np.asarray(
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
                ]
            )
            # Body links
            for i in range(self.options.morphology.n_links_body()):
                links[i] = Link.from_mesh(
                    name=self.convention.bodylink2name(i),
                    mesh="{}/salamander_body_{}.obj".format(
                        self.options.morphology.mesh_directory,
                        i
                    ),
                    pose=np.concatenate([
                        body_link_positions[i],
                        np.zeros(3)
                    ]),
                    scale=self.scale,
                    units=self.units,
                    color=[0.1, 0.7, 0.1, 1]
                )
            # Body joints
            for i in range(self.options.morphology.n_joints_body):
                joints[i] = Joint(
                    name=self.convention.bodyjoint2name(i),
                    joint_type="revolute",
                    parent=links[i],
                    child=links[i+1],
                    axis=[0, 0, 1],
                    limits=[-np.pi, np.pi, 1e10, 2*np.pi*100]
                )
        else:
            size = self.scale*np.array([0.08, 0.05, 0.04])
            shape_pose = [size[0]/2, 0, 0, 0, 0, 0]
            body_link_positions = np.zeros([
                self.options.morphology.n_links_body(), 3
            ])
            # Body links
            for i in range(self.options.morphology.n_links_body()):
                body_link_positions[i, 0] = i*size[0]
                links[i] = Link.box(
                    name=self.convention.bodylink2name(i),
                    size=size*self.scale,
                    pose=np.concatenate([
                        body_link_positions[i],
                        np.zeros(3)
                    ]),
                    inertial_pose=shape_pose,
                    shape_pose=shape_pose,
                    units=self.units,
                    color=[0.1, 0.7, 0.1, 1]
                )
            # Body joints
            for i in range(self.options.morphology.n_joints_body):
                joints[i] = Joint(
                    name=self.convention.bodyjoint2name(i),
                    joint_type="revolute",
                    parent=links[i],
                    child=links[i+1],
                    axis=[0, 0, 1],
                    limits=[-np.pi, np.pi, 1e10, 2*np.pi*100]
                )
        # Leg links
        leg_offset = self.scale*self.options.morphology.leg_offset
        leg_length = self.scale*self.options.morphology.leg_length
        leg_radius = self.scale*self.options.morphology.leg_radius
        for leg_i in range(self.options.morphology.n_legs//2):
            for side_i in range(2):
                sign = 1 if side_i else -1
                body_position = body_link_positions[
                    self.options.morphology.legs_parents[leg_i]+1
                ]
                # Shoulder 0
                pose = np.concatenate([
                    body_position +  [
                        0,
                        sign*leg_offset,
                        0
                    ],
                    [0, 0, 0]
                ])
                index = self.convention.leglink2index(
                    leg_i,
                    side_i,
                    0
                )+1
                links[index] = Link.sphere(
                    name=self.convention.leglink2name(
                        leg_i,
                        side_i,
                        0
                    ),
                    radius=1.1*leg_radius,
                    pose=pose,
                    units=self.units,
                    color=[0.7, 0.5, 0.5, 0.5]
                )
                links[index].inertial.mass = 0
                links[index].inertial.inertias = np.zeros(6)
                # Shoulder 1
                index = self.convention.leglink2index(
                    leg_i,
                    side_i,
                    1
                )+1
                links[index] = Link.sphere(
                    name=self.convention.leglink2name(
                        leg_i,
                        side_i,
                        1
                    ),
                    radius=1.3*leg_radius,
                    pose=pose,
                    units=self.units,
                    color=[0.9, 0.9, 0.9, 0.3]
                )
                links[index].inertial.mass = 0
                links[index].inertial.inertias = np.zeros(6)
                # Shoulder 2
                shape_pose = [
                    0, sign*(0.5*leg_length), 0,
                    np.pi/2, 0, 0
                ]
                links[self.convention.leglink2index(
                    leg_i,
                    side_i,
                    2
                )+1] = Link.capsule(
                    name=self.convention.leglink2name(
                        leg_i,
                        side_i,
                        2
                    ),
                    length=leg_length,
                    radius=leg_radius,
                    pose=pose,
                    inertial_pose=shape_pose,
                    shape_pose=shape_pose,
                    units=self.units
                )
                # Elbow
                pose = np.copy(pose)
                pose[1] += sign*leg_length
                links[self.convention.leglink2index(
                    leg_i,
                    side_i,
                    3
                )+1] = Link.capsule(
                    name=self.convention.leglink2name(
                        leg_i,
                        side_i,
                        3
                    ),
                    length=leg_length,
                    radius=leg_radius,
                    pose=pose,
                    inertial_pose=shape_pose,
                    shape_pose=shape_pose,
                    units=self.units,
                    # color=[
                    #     [[0.9, 0.0, 0.0, 1.0], [0.0, 0.9, 0.0, 1.0]],
                    #     [[0.0, 0.0, 0.9, 1.0], [1.0, 0.7, 0.0, 1.0]]
                    # ][leg_i][side_i]
                )
        # Leg joints
        for leg_i in range(self.options.morphology.n_legs//2):
            for side_i in range(2):
                for joint_i in range(self.options.morphology.n_dof_legs):
                    sign = 1 if side_i else -1
                    axis = [
                        [0, 0, sign],
                        [-sign, 0, 0],
                        [0, 1, 0],
                        [-sign, 0, 0]
                    ]
                    if joint_i == 0:
                        joints[self.convention.legjoint2index(
                            leg_i,
                            side_i,
                            joint_i
                        )] = Joint(
                            name=self.convention.legjoint2name(
                                leg_i,
                                side_i,
                                joint_i
                            ),
                            joint_type="revolute",
                            parent=links[5 if leg_i else 2],
                            child=links[self.convention.leglink2index(
                                leg_i,
                                side_i,
                                joint_i
                            )+1],
                            axis=axis[joint_i],
                            limits=[-np.pi, np.pi, 1e10, 2*np.pi*100]
                        )
                    else:
                        joints[self.convention.legjoint2index(
                            leg_i,
                            side_i,
                            joint_i
                        )] = Joint(
                            name=self.convention.legjoint2name(
                                leg_i,
                                side_i,
                                joint_i
                            ),
                            joint_type="revolute",
                            parent=links[self.convention.leglink2index(
                                leg_i,
                                side_i,
                                joint_i-1
                            )+1],
                            child=links[self.convention.leglink2index(
                                leg_i,
                                side_i,
                                joint_i
                            )+1],
                            axis=axis[joint_i],
                            limits=[-np.pi, np.pi, 1e10, 2*np.pi*100]
                        )

        # Create SDF
        sdf = ModelSDF(
            name="animat",
            pose=np.concatenate([
                np.asarray([0, 0, 0.1])*self.scale,
                [0, 0, 0]
            ]),
            links=links,
            joints=joints,
            units=self.units
        )
        sdf.write(filename="animat.sdf")
        import os
        print(os.getcwd() + "/animat.sdf")
        self._identity = pybullet.loadSDF(
            os.getcwd() + "/animat.sdf",
            useMaximalCoordinates=0,
            globalScaling=1
        )[0]
        # texUid = pybullet.loadTexture("/home/jonathan/Work/EPFL/PhD/Dev/FARMS/farms_bullet/farms_bullet/animats/amphibious/salamander_skin.jpg")
        # for i in range(self.options.morphology.n_links()):
        #     pybullet.changeVisualShape(
        #         self.identity,
        #         -1+i,
        #         textureUniqueId=texUid
        #     )
        # Joint order
        joints_names = [None for _ in range(self.options.morphology.n_joints())]
        joint_index = 0
        for joint_i in range(self.options.morphology.n_joints_body):
            joint_info = pybullet.getJointInfo(
                self.identity,
                joint_i
            )
            joints_names[joint_index] = joint_info[1].decode("UTF-8")
            joint_index += 1
        for leg_i in range(self.options.morphology.n_legs//2):
            for side_i in range(2):
                for part_i in range(self.options.morphology.n_dof_legs):
                    index = self.convention.leglink2index(
                        leg_i,
                        side_i,
                        part_i
                    )
                    joint_info = pybullet.getJointInfo(
                        self.identity,
                        index
                    )
                    joints_names[joint_index] = joint_info[1].decode("UTF-8")
                    joint_index += 1
        joints_names_dict = {
            name: i
            for i, name in enumerate(joints_names)
        }
        self.joints_order = [
            joints_names_dict[name]
            for name in [
                self.convention.bodyjoint2name(i)
                for i in range(self.options.morphology.n_joints_body)
            ] + [
                self.convention.legjoint2name(leg_i, side_i, joint_i)
                for leg_i in range(self.options.morphology.n_legs//2)
                for side_i in range(2)
                for joint_i in range(self.options.morphology.n_dof_legs)
            ]
        ]
        # Set names
        self.links['link_body_{}'.format(0)] = -1
        for i in range(self.options.morphology.n_links_body()-1):
            self.links['link_body_{}'.format(i+1)] = self.joints_order[i]
            self.joints['joint_link_body_{}'.format(i)] = self.joints_order[i]
        for leg_i in range(self.options.morphology.n_legs//2):
            for side in range(2):
                for joint_i in range(self.options.morphology.n_dof_legs):
                    self.links[
                        self.convention.leglink2name(
                            leg_i=leg_i,
                            side_i=side,
                            joint_i=joint_i
                        )
                    ] = self.joints_order[
                        self.convention.leglink2index(
                            leg_i=leg_i,
                            side_i=side,
                            joint_i=joint_i
                        )
                    ]
                    self.joints[
                        self.convention.legjoint2name(
                            leg_i=leg_i,
                            side_i=side,
                            joint_i=joint_i
                        )
                    ] = self.joints_order[
                        self.convention.legjoint2index(
                            leg_i=leg_i,
                            side_i=side,
                            joint_i=joint_i
                        )
                    ]
        if verbose:
            self.print_information()

    def spawn_body(self, verbose=False):
        """Spawn body"""
        if self.options.morphology.mesh_directory:
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
            body_shape = {
                "geometry": pybullet.GEOM_MESH,
                "size": [0, 0, 0],
                "color": body_color,
                "scale": [self.scale, self.scale, self.scale],
                "frame_position": [0, 0, 0]
            }
        else:
            body_link_positions = np.zeros([self.options.morphology.n_links_body(), 3])
            body_link_positions[0, 0] = self.scale*0.03
            body_link_positions[1:, 0] = self.scale*0.06
            body_shape = {
                "geometry": pybullet.GEOM_BOX,
                "size": self.scale*np.array([0.03, 0.02, 0.02]),
                "scale": [self.scale, self.scale, self.scale],
                "frame_position": self.scale*np.array([0.03, 0, 0])
            }
            # body_shape = {
            #     "geometry": pybullet.GEOM_CAPSULE,
            #     "radius": 0.03,
            #     "height": 0.06,
            #     "frame_orientation": [0, 0.5*np.pi, 0]
            # }
            # base_link = AnimatLink(
            #     self.units,
            #     geometry=pybullet.GEOM_MESH,
            #     filename="{}/amphibious_body_0.obj".format(self.options.morphology.mesh_directory),
            #     position=body_link_positions[0],
            #     joint_axis=[0, 0, 1],
            #     color=body_color,
            #     scale=[self.scale, self.scale, self.scale]
            # )
        if body_shape["geometry"] is pybullet.GEOM_MESH:
            body_shape["filename"] = (
                "{}/salamander_body_0.obj".format(self.options.morphology.mesh_directory)
            )
        base_link = AnimatLink(
            **body_shape,
            position=body_link_positions[0],
            joint_axis=[0, 0, 1],
            # color=body_color,
            units=self.units
        )
        links = [
            None
            for _ in range(self.options.morphology.n_links()-1)  # No base link
        ]
        for i in range(self.options.morphology.n_links_body()-1):
            if body_shape["geometry"] is pybullet.GEOM_MESH:
                body_shape["filename"] = (
                    "{}/salamander_body_{}.obj".format(
                        self.options.morphology.mesh_directory,
                        i+1
                    )
                )
            links[i] = AnimatLink(
                **body_shape,
                position=body_link_positions[i+1],
                parent=i,
                joint_axis=[0, 0, 1],
                # color=body_color
                units=self.units
            )
        # links = [None for _ in range(11)]
        # print("Creating amphibious body")
        # for link_i in range(11):
        #     links[link_i] = AnimatLink(
        #         self.units,
        #         geometry=pybullet.GEOM_MESH,
        #         filename="{}/amphibious_body_{}.obj".format(
        #             self.options.morphology.mesh_directory,
        #             link_i+1
        #         ),
        #         position=body_link_positions[link_i+1],
        #         parent=(
        #             links[link_i-1].collision
        #             if link_i > 0
        #             else 0
        #         ),
        #         joint_axis=[0, 0, 1],
        #         color=body_color,
        #         scale=[self.scale, self.scale, self.scale]
        #     )
        if self.options.morphology.n_legs:
            print("Creating animat legs")
        leg_offset = self.scale*self.options.morphology.leg_offset
        leg_length = self.scale*self.options.morphology.leg_length
        leg_radius = self.scale*self.options.morphology.leg_radius
        for leg_i in range(self.options.morphology.n_legs//2):
            for side in range(2):
                sign = 1 if side else -1
                offset = np.zeros(3)
                offset[0] = body_shape["size"][0]
                offset[1] = sign*leg_offset
                position = np.zeros(3)
                position[1] = 0.5*sign*leg_length
                # Shoulder1
                links[
                    self.convention.leglink2index(leg_i, side, 0)
                ] = AnimatLink(
                    self.units,
                    geometry=pybullet.GEOM_SPHERE,
                    radius=1.2*leg_radius,
                    position=offset,
                    parent=(
                        links[
                            self.options.morphology.legs_parents[leg_i]
                        ].collision
                    ),  # Different orders seem to change nothing
                    joint_axis=[0, 0, sign],
                    mass=0,
                    color=[0.9, 0.0, 0.0, 0.3]
                )
                # Shoulder2
                links[
                    self.convention.leglink2index(leg_i, side, 1)
                ] = AnimatLink(
                    self.units,
                    geometry=pybullet.GEOM_SPHERE,
                    radius=1.5*leg_radius,
                    parent=links[
                        self.convention.leglink2index(leg_i, side, 0)
                    ].collision,
                    joint_axis=[-sign, 0, 0],
                    mass=0,
                    color=[0.9, 0.9, 0.9, 0.3]
                )
                # Upper leg
                links[
                    self.convention.leglink2index(leg_i, side, 2)
                ] = AnimatLink(
                    self.units,
                    geometry=pybullet.GEOM_CAPSULE,
                    radius=leg_radius,
                    height=leg_length,
                    frame_position=position,
                    frame_orientation=[np.pi/2, 0, 0],
                    parent=links[
                        self.convention.leglink2index(leg_i, side, 1)
                    ].collision,
                    joint_axis=[0, 1, 0]
                )
                # Lower leg
                links[
                    self.convention.leglink2index(leg_i, side, 3)
                ] = AnimatLink(
                    self.units,
                    geometry=pybullet.GEOM_CAPSULE,
                    radius=leg_radius,
                    height=leg_length,
                    position=2*position,
                    frame_position=position,
                    frame_orientation=[np.pi/2, 0, 0],
                    parent=links[
                        self.convention.leglink2index(leg_i, side, 2)
                    ].collision,
                    joint_axis=[-sign, 0, 0],
                    **(
                        {
                            "color": [
                                [[0.9, 0.0, 0.0, 1.0], [0.0, 0.9, 0.0, 1.0]],
                                [[0.0, 0.0, 0.9, 1.0], [1.0, 0.7, 0.0, 1.0]]
                            ][leg_i][side]
                        }
                        if self.options.morphology.n_legs < 3
                        else {}
                    )
                )
        for link_i, link in enumerate(links):
            assert link is not None, "link {} is None".format(link_i)

        if verbose:
            for link_i, link in enumerate(links):
                print(" {} (parent={}): {} (visual={}, collision={})".format(
                    link_i+1,
                    link.parent,
                    link.position,
                    link.visual,
                    link.collision
                ))
        self._identity = pybullet.createMultiBody(
            baseMass=base_link.mass*self.units.kilograms,
            baseCollisionShapeIndex=base_link.collision,
            baseVisualShapeIndex=base_link.visual,
            basePosition=np.array(self.options.spawn.position)*self.units.meters,
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
        # Joint order
        joints_names = [None for _ in range(self.options.morphology.n_joints())]
        joint_index = 0
        for joint_i in range(self.options.morphology.n_joints_body):
            joint_info = pybullet.getJointInfo(
                self.identity,
                joint_i
            )
            joints_names[joint_index] = joint_info[1].decode("UTF-8")
            joint_index += 1
        for leg_i in range(self.options.morphology.n_legs//2):
            for side_i in range(2):
                for part_i in range(self.options.morphology.n_dof_legs):
                    index = self.convention.leglink2index(
                        leg_i,
                        side_i,
                        part_i
                    )
                    joint_info = pybullet.getJointInfo(
                        self.identity,
                        index
                    )
                    joints_names[joint_index] = joint_info[1].decode("UTF-8")
                    joint_index += 1
        self.joints_order = np.argsort([
            int(name.replace("joint", ""))
            for name in joints_names
        ])
        # Set names
        self.links['link_body_{}'.format(0)] = -1
        for i in range(self.options.morphology.n_links_body()-1):
            self.links['link_body_{}'.format(i+1)] = self.joints_order[i]
            self.joints['joint_link_body_{}'.format(i)] = self.joints_order[i]
        for leg_i in range(self.options.morphology.n_legs//2):
            for side in range(2):
                for joint_i in range(self.options.morphology.n_dof_legs):
                    self.links[
                        self.convention.leglink2name(
                            leg_i=leg_i,
                            side_i=side,
                            joint_i=joint_i
                        )
                    ] = self.joints_order[
                        self.convention.leglink2index(
                            leg_i=leg_i,
                            side_i=side,
                            joint_i=joint_i
                        )
                    ]
                    self.joints[
                        self.convention.legjoint2name(
                            leg_i=leg_i,
                            side_i=side,
                            joint_i=joint_i
                        )
                    ] = self.joints_order[
                        self.convention.legjoint2index(
                            leg_i=leg_i,
                            side_i=side,
                            joint_i=joint_i
                        )
                    ]
        if verbose:
            self.print_information()

    # @classmethod
    # def spawn_sdf(cls, iterations, timestep, gait="walking", **kwargs):
    #     """Spawn amphibious"""
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
                self.joints_order,
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
            for i in range(self.options.morphology.n_links_body())
        ] + [
            [
                "link_leg_{}_{}_{}".format(leg_i, side, joint_i),
                # 12 + leg_i*2*4 + side_i*4 + joint_i,
                self.convention.leglink2index(
                    leg_i,
                    side_i,
                    joint_i
                )+1,
                self.links["link_leg_{}_{}_{}".format(
                    leg_i,
                    side,
                    joint_i,
                    n_body_joints=self.options.morphology.n_joints_body
                )]
            ]
            for leg_i in range(self.options.morphology.n_legs//2)
            for side_i, side in enumerate(["L", "R"])
            for joint_i in range(self.options.morphology.n_dof_legs)
        ]
        self.sensors.add({
            "links": AmphibiousGPS(
                array=self.data.sensors.gps.array,
                animat_id=self.identity,
                links=links,
                options=self.options,
                units=self.units
            )
        })

    def set_body_properties(self):
        """Set body properties"""
        # Masses
        for i in range(self.options.morphology.n_links()):
            self.masses[i] = pybullet.getDynamicsInfo(self.identity, i-1)[0]
        # Deactivate collisions
        links_no_collisions = [
            "link_body_{}".format(body_i)
            for body_i in range(0)
        ] + [
            "link_leg_{}_{}_{}".format(leg_i, side, joint_i)
            for leg_i in range(self.options.morphology.n_legs//2)
            for side in ["L", "R"]
            for joint_i in range(self.options.morphology.n_dof_legs-1)
        ]
        self.set_collisions(links_no_collisions, group=0, mask=0)
        # Deactivate damping
        links_no_damping = [
            "link_body_{}".format(body_i)
            for body_i in range(self.options.morphology.n_links_body())
        ] + [
            "link_leg_{}_{}_{}".format(leg_i, side, joint_i)
            for leg_i in range(self.options.morphology.n_legs//2)
            for side in ["L", "R"]
            for joint_i in range(self.options.morphology.n_dof_legs)
        ]
        small = 0
        self.set_links_dynamics(
            links_no_damping,
            linearDamping=small,
            angularDamping=small,
            jointDamping=small
        )
        # Friction
        self.set_links_dynamics(
            self.links,
            lateralFriction=0.5,
            spinningFriction=small,
            rollingFriction=small,
        )
        self.set_links_dynamics(
            self.feet_names,
            lateralFriction=0.7,
            spinningFriction=small,
            rollingFriction=small,
            # contactStiffness=1e3,
            # contactDamping=1e6
        )

    def setup_controller(self):
        """Setup controller"""
        self.controller = AmphibiousController.from_data(
            self.identity,
            animat_options=self.options,
            animat_data=self.data,
            timestep=self.timestep,
            joints_order=self.joints_order,
            units=self.units
        )

    def viscous_swimming_forces(self, iteration, water_surface):
        """Animat swimming physics"""
        viscous_forces(
            iteration,
            self.data.sensors.gps,
            self.data.sensors.hydrodynamics.array,
            [
                link_i
                for link_i in range(self.options.morphology.n_links_body())
                if (
                    self.data.sensors.gps.com_position(iteration, link_i)[2]
                    < water_surface
                )
            ],
            masses=self.masses,
            coefficients=[
                self.options.morphology.scale**3*np.array([-1e-1, -1e0, -1e0]),
                self.options.morphology.scale**6*np.array([-1e-3, -1e-3, -1e-3])
            ]
        )

    def apply_swimming_forces(
            self, iteration, water_surface, link_frame=True, debug=False
    ):
        """Animat swimming physics"""
        swimming_motion(
            iteration,
            self.data.sensors.hydrodynamics.array,
            self.identity,
            [
                [i, self.links["link_body_{}".format(i)]]
                for i in range(self.options.morphology.n_links_body())
                if (
                    self.data.sensors.gps.com_position(iteration, i)[2]
                    < water_surface
                )
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
                    for i in range(self.options.morphology.n_links_body())
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
