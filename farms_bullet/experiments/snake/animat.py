"""Snake"""

import os

import numpy as np
import pybullet

from ...animats.animat import Animat
from ...animats.link import AnimatLink
from ...plugins.swimming import viscous_swimming
from ...sensors.sensors import (
    Sensors,
    JointsStatesSensor,
    ContactsSensors
)
from ..salamander.animat_data import (
    SalamanderOscillatorNetworkState,
    SalamanderData
)
from ..salamander.control import SalamanderController
from ..salamander.sensors import SalamanderGPS


class Snake(Animat):
    """Snake animat"""

    def __init__(self, options, timestep, iterations, units):
        super(Snake, self).__init__(options=options)
        self.timestep = timestep
        self.n_iterations = iterations
        self.joints_order = None
        self.data = SalamanderData.from_options(
            SalamanderOscillatorNetworkState.default_state(iterations, options),
            options,
            iterations
        )
        # Hydrodynamic forces
        self.hydrodynamics = None
        # Sensors
        self.sensors = Sensors()
        # Physics
        self.units = units
        self.scale = options.morphology.scale

    def spawn(self):
        """Spawn snake"""
        self.spawn_body()
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

    def spawn_body(self):
        """Spawn body"""
        # meshes_directory = (
        #     "/{}/../salamander/meshes".format(
        #         os.path.dirname(os.path.realpath(__file__))
        #     )
        # )
        # body_link_positions = self.scale*np.diff(
        #     [  # From SDF
        #         [0, 0, 0],
        #         [0.200000003, 0, 0.0069946074],
        #         [0.2700000107, 0, 0.010382493],
        #         [0.3400000036, 0, 0.0106022889],
        #         [0.4099999964, 0, 0.010412137],
        #         [0.4799999893, 0, 0.0086611426],
        #         [0.5500000119, 0, 0.0043904358],
        #         [0.6200000048, 0, 0.0006898994],
        #         [0.6899999976, 0, 8.0787e-06],
        #         [0.7599999905, 0, -4.89001e-05],
        #         [0.8299999833, 0, 0.0001386079],
        #         [0.8999999762, 0, 0.0003494423]
        #     ],
        #     axis=0,
        #     prepend=0
        # )
        # body_color = [0, 0.3, 0, 1]
        body_link_positions = np.zeros([self.options.morphology.n_links_body(), 3])
        body_link_positions[1:, 0] = 0.06
        body_shape = {
            "geometry": pybullet.GEOM_BOX,
            "size": [0.03, 0.02, 0.02]
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
        #     filename="{}/salamander_body_0.obj".format(meshes_directory),
        #     position=body_link_positions[0],
        #     joint_axis=[0, 0, 1],
        #     color=body_color,
        #     scale=[self.scale, self.scale, self.scale]
        # )
        base_link = AnimatLink(
            **body_shape,
            position=body_link_positions[0],
            joint_axis=[0, 0, 1],
            # color=body_color,
            units=self.units
        )
        links = [
            AnimatLink(
                **body_shape,
                position=body_link_positions[i+1],
                parent=i,
                joint_axis=[0, 0, 1],
                # color=body_color
                scale=[self.scale, self.scale, self.scale],
                units=self.units
            )
            for i in range(self.options.morphology.n_links_body()-1)
        ]
        # links = [None for _ in range(11)]
        # print("Creating snake body")
        # for link_i in range(11):
        #     links[link_i] = AnimatLink(
        #         self.units,
        #         geometry=pybullet.GEOM_MESH,
        #         filename="{}/salamander_body_{}.obj".format(
        #             meshes_directory,
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
        index = 10
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
            basePosition=np.array([0, 0, 0.1])*self.units.meters,
            baseOrientation=pybullet.getQuaternionFromEuler([0, 0, 0]),
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
        joints_names = [None for _ in range(self.options.morphology.n_joints_body)]
        joints_order = [None for _ in range(self.options.morphology.n_joints_body)]
        joint_index = 0
        for joint_i in range(self.options.morphology.n_joints_body):
            joint_info = pybullet.getJointInfo(
                self.identity,
                joint_i
            )
            print("{}: {}".format(index, joint_info))
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
        self.print_information()

    # @classmethod
    # def spawn_sdf(cls, iterations, timestep, gait="walking", **kwargs):
    #     """Spawn snake"""
    #     return cls.from_sdf(
    #         "{}/.farms/models/biorob_snake/model.sdf".format(
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
        # self.sensors.add({
        #     "contacts": ContactsSensors(
        #         self.data.sensors.contacts.array,
        #         [self._identity for _ in self.feet_names],
        #         [self.links[foot] for foot in self.feet_names],
        #         self.units.newtons
        #     )
        # })
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
        ]
        self.set_collisions(links_no_collisions, group=0, mask=0)
        # Deactivate damping
        links_no_damping = [
            "link_body_{}".format(body_i)
            for body_i in range(self.options.morphology.n_links_body())
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
            lateralFriction=1e-2,
            spinningFriction=small,
            rollingFriction=small,
        )
        # self.set_links_dynamics(
        #     self.feet_names,
        #     lateralFriction=0.7,
        #     spinningFriction=small,
        #     rollingFriction=small,
        #     # contactStiffness=1e3,
        #     # contactDamping=1e6
        # )

    def setup_controller(self):
        """Setup controller"""
        self.controller = SalamanderController.from_data(
            self.identity,
            animat_options=self.options,
            animat_data=self.data,
            timestep=self.timestep,
            joints_order=self.joints_order,
            units=self.units
        )

    def animat_swimming_physics(self, iteration):
        """Animat swimming physics"""
        viscous_swimming(
            iteration,
            self.data.sensors.gps,
            self.data.sensors.hydrodynamics.array,
            self.identity,
            [
                [i, self.links["link_body_{}".format(i)]]
                for i in range(self.options.morphology.n_links_body())
            ],
            coefficients=[
                self.options.morphology.scale**3*np.array([-1e-1, -1e0, -1e0]),
                self.options.morphology.scale**6*np.array([-1e-3, -1e-3, -1e-3])
            ],
            units=self.units
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
