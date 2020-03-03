"""Amphibious"""

import re
import numpy as np
import pybullet

from ...animats.animat import Animat
from ...plugins.swimming import (
    viscous_forces,
    resistive_forces,
    swimming_motion,
    swimming_debug
)
from ...sensors.sensors import (
    Sensors,
    JointsStatesSensor,
    ContactsSensors
)

from farms_sdf.sdf import ModelSDF, Link, Joint

from .convention import AmphibiousConvention
from .animat_data import (
    AmphibiousOscillatorNetworkState,
    AmphibiousData
)
from .control import AmphibiousController
from .sensors import AmphibiousGPS


def links_ordering(text):
    """links ordering"""
    text = re.sub("version[0-9]_", "", text)
    text = re.sub("[a-z]", "", text)
    text = re.sub("_", "", text)
    text = int(text)
    return [text]


def initial_pose(identity, spawn_options, units):
    """Initial pose"""
    pybullet.resetBasePositionAndOrientation(
        identity,
        spawn_options.position,
        pybullet.getQuaternionFromEuler(
            spawn_options.orientation
        )
    )
    pybullet.resetBaseVelocity(
        objectUniqueId=identity,
        linearVelocity=np.array(spawn_options.velocity_lin)*units.velocity,
        angularVelocity=np.array(spawn_options.velocity_ang)/units.seconds
    )
    # print(spawn_options.velocity_lin)
    # print(spawn_options.velocity_ang)
    # raise Exception
    if (
            spawn_options.joints_positions is not None
            or spawn_options.joints_velocities is not None
    ):
        if spawn_options.joints_positions is None:
            spawn_options.joints_positions = np.zeros_like(
                spawn_options.joints_velocities
            )
        if spawn_options.joints_velocities is None:
            spawn_options.joints_velocities = np.zeros_like(
                spawn_options.joints_positions
            )
        for joint_i, (position, velocity) in enumerate(zip(
                spawn_options.joints_positions,
                spawn_options.joints_velocities
        )):
            pybullet.resetJointState(
                bodyUniqueId=identity,
                jointIndex=joint_i,
                targetValue=position,
                targetVelocity=velocity/units.seconds
            )


class Amphibious(Animat):
    """Amphibious animat"""

    def __init__(self, options, timestep, iterations, units, sdf=None):
        super(Amphibious, self).__init__(options=options)
        self.sdf = sdf
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
        # Spawn
        self.spawn_sdf()
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
        if self.sdf:
            self._identity = pybullet.loadSDF(
                self.sdf,
                useMaximalCoordinates=0,
                globalScaling=1
            )[0]
            initial_pose(self._identity, self.options.spawn, self.units)
        else:
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
                        xyz=[0, 0, 1],
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
                        # inertial_pose=shape_pose,
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
                        xyz=[0, 0, 1],
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
                        # inertial_pose=shape_pose,
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
                        # inertial_pose=shape_pose,
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
                                xyz=axis[joint_i],
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
                                xyz=axis[joint_i],
                                limits=[-np.pi, np.pi, 1e10, 2*np.pi*100]
                            )

            # Use 2D
            use_2d = False
            constraint_links = [
                Link.empty(
                    name="world",
                    pose=[0, 0, 0, 0, 0, 0],
                    units=self.units
                ),
                Link.empty(
                    name="world_2",
                    pose=[0, 0, 0, 0, 0, 0],
                    units=self.units
                )
            ] if use_2d else []
            constraint_joints = [
                Joint(
                    name="world_joint",
                    joint_type="prismatic",
                    parent=constraint_links[0],
                    child=constraint_links[1],
                    pose=[0, 0, 0, 0, 0, 0],
                    xyz=[1, 0, 0],
                    limits=np.array([-1, 1, 0, 1])
                ),
                Joint(
                    name="world_joint2",
                    joint_type="prismatic",
                    parent=constraint_links[1],
                    child=links[0],
                    pose=[0, 0, 0, 0, 0, 0],
                    xyz=[0, 0, 1],
                    limits=np.array([-1, 1, 0, 1])
                )
            ] if use_2d else []

            # Create SDF
            sdf = ModelSDF(
                name="animat",
                pose=np.concatenate([
                    np.asarray([0, 0, 0.1])*self.scale,
                    [0, 0, 0]
                ]),
                links=constraint_links+links,
                joints=constraint_joints+joints,
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
            initial_pose(self._identity, self.options.spawn, self.units)
            # texUid = pybullet.loadTexture("/home/jonathan/Work/EPFL/PhD/Dev/FARMS/farms_bullet/farms_bullet/animats/amphibious/salamander_skin.jpg")
            # for i in range(self.options.morphology.n_links()):
            #     pybullet.changeVisualShape(
            #         self.identity,
            #         -1+i,
            #         textureUniqueId=texUid
            #     )
            # Joint order
        n_joints = pybullet.getNumJoints(self.identity)
        print(n_joints)
        joints_names = [None for _ in range(n_joints)]
        joint_index = 0
        for joint_i in range(n_joints):
            joint_info = pybullet.getJointInfo(
                self.identity,
                joint_i
            )
            joints_names[joint_index] = joint_info[1].decode("UTF-8")
            joint_index += 1
        if self.sdf:
            joints_names_dict = {
                name: i
                for i, name in enumerate(joints_names)
            }
            self.joints_order = [
                joints_names_dict[name]
                for name in sorted(joints_names_dict.keys(), key=links_ordering)
            ]
            # Set names
            self.links['link_body_{}'.format(0)] = -1
            for i in range(self.options.morphology.n_links_body()-1):
                self.links['link_body_{}'.format(i+1)] = self.joints_order[i]
                self.joints['joint_link_body_{}'.format(i)] = self.joints_order[i]
        else:
            for joint_index in range(n_joints):
                joint_info = pybullet.getJointInfo(
                    self.identity,
                    joint_index
                )
                joints_names[joint_index] = joint_info[1].decode("UTF-8")
                # joint_index += 1
            joints_names_dict = {
                name: i
                for i, name in enumerate(joints_names)
            }
            print(joints_names_dict)
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
        if self.options.control.kinematics_file:
            self.controller = AmphibiousController.from_kinematics(
                self.identity,
                animat_options=self.options,
                animat_data=self.data,
                timestep=self.timestep,
                joints_order=self.joints_order,
                units=self.units
            )
        else:
            self.controller = AmphibiousController.from_data(
                self.identity,
                animat_options=self.options,
                animat_data=self.data,
                timestep=self.timestep,
                joints_order=self.joints_order,
                units=self.units
            )

    def viscous_swimming_forces(self, iteration, water_surface, **kwargs):
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
            **kwargs
        )

    def resistive_swimming_forces(self, iteration, water_surface, **kwargs):
        """Animat swimming physics"""
        resistive_forces(
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
            **kwargs
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
