"""Arena"""

import os

import numpy as np
import pybullet

from farms_sdf.sdf import ModelSDF, Link, Joint
from farms_models.utils import get_sdf_path

from .create import create_scene
from ..simulations.element import SimulationElement
# from ..animats.link import AnimatLink


# class Floor(SimulationElement):
#     """Floor"""

#     def __init__(self, position):
#         super(Floor, self).__init__()
#         self._position = np.array(position)

#     def spawn(self):
#         """Spawn floor"""
#         size = 0.5*np.array([10, 10, 10])
#         base_link = AnimatLink(
#             geometry=pybullet.GEOM_BOX,
#             size=size,
#             inertial_position=[0, 0, 0],
#             position=[0, 0, 0],
#             joint_axis=[0, 0, 1],
#             mass=0,
#             color=[1, 0, 0, 1],
#             # collision_options=collision_options,
#             # visual_options=visual_options
#         )
#         self._identity = pybullet.createMultiBody(
#             baseMass=base_link.mass,
#             baseCollisionShapeIndex=base_link.collision,
#             baseVisualShapeIndex=base_link.visual,
#             basePosition=self._position-np.array([0, 0, size[2]]),
#             baseOrientation=pybullet.getQuaternionFromEuler([0, 0, 0]),
#             baseInertialFramePosition=base_link.inertial_position,
#             baseInertialFrameOrientation=base_link.inertial_orientation
#         )
#         dir_path = os.path.dirname(os.path.realpath(__file__))
#         texUid = pybullet.loadTexture(dir_path+"/BIOROB2_blue.png")
#         pybullet.changeVisualShape(
#             self._identity, -1,
#             textureUniqueId=texUid,
#             # rgbaColor=[1, 1, 1, 1],
#             # specularColor=[1, 1, 1]
#         )


class FloorURDF(SimulationElement):
    """Floor"""

    def __init__(self, position):
        super(FloorURDF, self).__init__()
        self._position = position

    def spawn(self):
        """Spawn floor"""
        self._identity = self.from_urdf(
            "plane.urdf",
            basePosition=self._position
        )
        dir_path = os.path.dirname(os.path.realpath(__file__))
        texUid = pybullet.loadTexture(dir_path+"/BIOROB2_blue.png")
        pybullet.changeVisualShape(
            self._identity, -1,
            textureUniqueId=texUid,
            rgbaColor=[1, 1, 1, 1],
            specularColor=[1, 1, 1]
        )


class Arena:
    """Documentation for Arena"""

    def __init__(self, elements):
        super(Arena, self).__init__()
        self.elements = elements
        self.water_surface = -np.inf

    def spawn(self):
        """Spawn"""
        for element in self.elements:
            element.spawn()


class FlooredArena(Arena):
    """Arena with floor"""

    def __init__(self, position=None):
        super(FlooredArena, self).__init__(
            [FloorURDF(position if position is not None else [0, 0, 0])]
        )

    @property
    def floor(self):
        """Floor"""
        return self.elements[0]


class ArenaScaffold(FlooredArena):
    """Arena for scaffolding"""

    def spawn(self):
        """Spawn"""
        FlooredArena.spawn(self)
        create_scene(self.floor.identity)


# class Ramp(SimulationElement):
#     """Floor"""

#     def __init__(self, angle, units):
#         super(Ramp, self).__init__()
#         self.angle = angle
#         self.units = units

#     def spawn(self):
#         """Spawn floor"""
#         ground_dim = [1, 20, 0.1]
#         ramp_dim = [3, 20, 0.1]
#         upper_lower_dim = [1, 20, 0.1]
#         arena_color = [1, 0.8, 0.5, 1.0]

#         # Arena definition
#         base_link = AnimatLink(
#             geometry=pybullet.GEOM_BOX,
#             size=ground_dim,
#             mass=0,
#             joint_axis=[0, 0, 1],
#             color=arena_color,
#             units=self.units
#         )
#         links = [
#             AnimatLink(
#                 geometry=pybullet.GEOM_BOX,
#                 size=ramp_dim,
#                 mass=0,
#                 parent=0,
#                 frame_position=[
#                     (
#                         - ground_dim[0]
#                         - np.cos(self.angle) * ramp_dim[0]
#                     ),
#                     0,
#                     np.sin(self.angle) * ramp_dim[0]
#                 ],
#                 frame_orientation=[0, self.angle, 0],
#                 joint_axis=[0, 0, 1],
#                 color=arena_color,
#                 units=self.units
#             ),
#             AnimatLink(
#                 geometry=pybullet.GEOM_BOX,
#                 size=ground_dim,
#                 mass=0,
#                 parent=1,
#                 frame_position=[
#                     (
#                         - ground_dim[0]
#                         - 2 * np.cos(self.angle) * ramp_dim[0]
#                         - upper_lower_dim[0]
#                     ),
#                     0,
#                     2 * np.sin(self.angle) * ramp_dim[0]
#                 ],
#                 frame_orientation=[0, 0, 0],
#                 joint_axis=[0, 0, 1],
#                 color=arena_color,
#                 units=self.units
#             )
#         ]

#         # Spawn
#         self._identity = pybullet.createMultiBody(
#             baseMass=base_link.mass,
#             baseCollisionShapeIndex=base_link.collision,
#             baseVisualShapeIndex=base_link.visual,
#             basePosition=[0, 0, -ground_dim[2]],
#             baseOrientation=pybullet.getQuaternionFromEuler([0, 0, 0]),
#             linkMasses=[link.mass for link in links],
#             linkCollisionShapeIndices=[link.collision for link in links],
#             linkVisualShapeIndices=[link.visual for link in links],
#             linkPositions=[link.position for link in links],
#             linkOrientations=[link.orientation for link in links],
#             linkInertialFramePositions=[
#                 link.inertial_position
#                 for link in links
#             ],
#             linkInertialFrameOrientations=[
#                 link.inertial_orientation
#                 for link in links
#             ],
#             linkParentIndices=[link.parent for link in links],
#             linkJointTypes=[link.joint_type for link in links],
#             linkJointAxis=[link.joint_axis for link in links]
#         )

#         # Textures
#         texture_file = "{}/BIOROB2_blue.png".format(
#             os.path.dirname(os.path.realpath(__file__))
#         )
#         texUid = pybullet.loadTexture(texture_file)
#         for i in range(3):
#             pybullet.changeVisualShape(
#                 self._identity, -1+i, textureUniqueId=texUid
#             )

#         # Dynamics properties
#         pybullet.changeDynamics(
#             bodyUniqueId=self.identity,
#             linkIndex=0,
#             lateralFriction=2,
#             spinningFriction=0,
#             rollingFriction=0,
#             contactDamping=1e2,
#             contactStiffness=1e4
#         )

#         # # Texture
#         # dir_path = os.path.dirname(os.path.realpath(__file__))
#         # texUid = pybullet.loadTexture(dir_path+"/BIOROB2_blue.png")
#         # pybullet.changeVisualShape(
#         #     self._identity, -1,
#         #     textureUniqueId=texUid,
#         #     rgbaColor=[1, 1, 1, 1],
#         #     specularColor=[1, 1, 1]
#         # )


class RampSDF(SimulationElement):
    """Floor"""

    def __init__(self, angle, units):
        super(RampSDF, self).__init__()
        self.angle = angle
        self.units = units

    def spawn(self, sdf=True):
        """Spawn floor"""
        if sdf:
            sdf = get_sdf_path(
                name='arena_ramp',
                version='angle_{}_texture'.format(int(np.rad2deg(self.angle)))
            )
            print(sdf)
            self._identity = pybullet.loadSDF(
                sdf,
                useMaximalCoordinates=0,
                globalScaling=1
            )[0]
            # Texture
            dir_path = os.path.join(os.path.dirname(sdf), 'meshes')
            path = dir_path+"/BIOROB2_blue.png"
            print(path)
        else:
            ground_dim = [2, 20, 0.1]
            ramp_dim = [6, 20, 0.1]
            upper_lower_dim = [1, 20, 0.1]
            # arena_color = [1, 0.8, 0.5, 1.0]
            arena_color = [1, 1.0, 1.0, 1.0]

            # Arena definition
            links = [None for _ in range(3)]
            # links[0] = Link.box(
            #     name="floor_0",
            #     size=ground_dim,
            #     pose=[0, 0, 0, 0, 0, 0],
            #     shape_pose=[0, 0, -0.5*ground_dim[2], 0, 0, 0],
            #     units=self.units,
            #     color=arena_color
            # )
            dir_path = os.path.dirname(os.path.realpath(__file__))
            links[0] = Link.from_mesh(
                name="floor_0",
                mesh="{}/arena.obj".format(dir_path),
                pose=[0, 0, 0, 0, 0, 0],
                scale=1,
                shape_pose=[0, 0, -0.5*ground_dim[2], np.pi/2, 0, 0],
                units=self.units,
                color=arena_color
            )
            links[0].inertial.mass = 0
            links[0].inertial.inertia = np.zeros(6)
            # links[1] = Link.box(
            #     name="floor_1",
            #     size=ramp_dim,
            #     pose=[
            #         (
            #             -0.5*(ground_dim[0]+ramp_dim[0])
            #             + 0.5*(1-np.cos(self.angle))*ramp_dim[0]
            #         ),
            #         0,
            #         0.5*ramp_dim[0]*np.sin(self.angle),
            #         0,
            #         self.angle,
            #         0
            #     ],
            #     shape_pose=[0, 0, -0.5*ground_dim[2], 0, 0, 0],
            #     units=self.units,
            #     color=arena_color
            # )
            links[1] = Link.from_mesh(
                name="floor_1",
                mesh="{}/arena_ramp.obj".format(dir_path),
                pose=[
                    (
                        -0.5*(ground_dim[0]+ramp_dim[0])
                        + 0.5*(1-np.cos(self.angle))*ramp_dim[0]
                    ),
                    0,
                    0.5*ramp_dim[0]*np.sin(self.angle),
                    0,
                    self.angle,
                    0
                ],
                scale=1,
                shape_pose=[0, 0, -0.5*ground_dim[2], np.pi/2, 0, 0],
                units=self.units,
                color=arena_color
            )
            links[1].inertial.mass = 0
            links[1].inertial.inertia = np.zeros(6)
            # links[2] = Link.box(
            #     name="floor_2",
            #     size=ground_dim,
            #     pose=[
            #         -ground_dim[0]-ramp_dim[0] + (1-np.cos(self.angle))*ramp_dim[0],
            #         0,
            #         ramp_dim[0]*np.sin(self.angle),
            #         0,
            #         0,
            #         0
            #     ],
            #     shape_pose=[0, 0, -0.5*ground_dim[2], 0, 0, 0],
            #     units=self.units,
            #     color=arena_color
            # )
            links[2] = Link.from_mesh(
                name="floor_2",
                mesh="{}/arena.obj".format(dir_path),
                pose=[
                    -ground_dim[0]-ramp_dim[0] + (1-np.cos(self.angle))*ramp_dim[0],
                    0,
                    ramp_dim[0]*np.sin(self.angle),
                    0,
                    0,
                    0
                ],
                scale=1,
                shape_pose=[0, 0, -0.5*ground_dim[2], np.pi/2, 0, 0],
                units=self.units,
                color=arena_color
            )
            links[2].inertial.mass = 0
            links[2].inertial.inertia = np.zeros(6)
            # Joints
            joints = [None, None]
            for i in range(2):
                joints[i] = Joint(
                    name="joint_{}".format(i),
                    joint_type="revolute",
                    parent=links[i],
                    child=links[i+1],
                    axis=[0, 1, 0],
                    limits=[-np.pi, np.pi, 1e10, 2*np.pi*100]
                )

            # Spawn
            sdf = ModelSDF(
                name="arena",
                pose=np.zeros(6),
                links=links,
                joints=joints,
                units=self.units
            )
            sdf.write(filename="arena.sdf")
            print(os.getcwd() + "/arena.sdf")
            self._identity = pybullet.loadSDF(
                os.getcwd() + "/arena.sdf",
                useMaximalCoordinates=0,
                globalScaling=1
            )[0]

            # # Textures
            # texture_file = "{}/BIOROB2_blue.png".format(
            #     os.path.dirname(os.path.realpath(__file__))
            # )
            # texUid = pybullet.loadTexture(texture_file)
            # for i in range(3):
            #     pybullet.changeVisualShape(
            #         self._identity, -1+i, textureUniqueId=texUid
            #     )
            # Texture
            dir_path = os.path.dirname(os.path.realpath(__file__))
            path = dir_path+"/BIOROB2_blue.png"
            print(path)

        texture = pybullet.loadTexture(path)
        for i in range(3):
            pybullet.changeVisualShape(
                objectUniqueId=self.identity,
                linkIndex=-1+i,
                shapeIndex=-1,
                textureUniqueId=texture,
                # rgbaColor=[1, 1, 1, 1],
                # specularColor=[1, 1, 1]
            )

        # Dynamics properties
        pybullet.changeDynamics(
            bodyUniqueId=self.identity,
            linkIndex=0,
            lateralFriction=2,
            spinningFriction=0,
            rollingFriction=0,
            # contactDamping=1e2,
            # contactStiffness=1e4
        )


class ArenaRamp(Arena):
    """ArenaRamp"""

    def __init__(self, units, ramp_angle=None, elements=None):
        angle = (
            np.deg2rad(ramp_angle)
            if ramp_angle is not None
            else np.deg2rad(30)
        )
        if elements is None:
            elements = []
        super(ArenaRamp, self).__init__(
            [RampSDF(angle=angle, units=units)]
            + elements
        )

    @property
    def floor(self):
        """Floor"""
        return self.elements[0]


class Water(SimulationElement):
    """Floor"""

    def __init__(self, units, water_surface=-0.1):
        super(Water, self).__init__()
        self.units = units
        self.water_surface = water_surface

    def spawn(self, sdf=True):
        """Spawn floor"""
        if sdf:
            sdf = get_sdf_path(name='arena_water', version='v0')
            print(sdf)
            self._identity = pybullet.loadSDF(
                sdf,
                useMaximalCoordinates=0,
                globalScaling=1
            )[0]
            pos = pybullet.getBasePositionAndOrientation(
                bodyUniqueId=self._identity
            )[0]
            pybullet.resetBasePositionAndOrientation(
                bodyUniqueId=self._identity,
                posObj=np.array(pos) + np.array([0, 0, self.water_surface]),
                ornObj=[0, 0, 0, 1],
            )
        else:
            water_size = [50, 50, 50]
            water_color = [0.5, 0.5, 0.9, 0.7]

            # Water definition
            link = Link.box(
                name="water",
                size=water_size,
                pose=[0, 0, 0, 0, 0, 0],
                shape_pose=[0, 0, self.water_surface-0.5*water_size[2], 0, 0, 0],
                units=self.units,
                color=water_color
            )
            link.collisions = []
            link.inertial.mass = 0
            link.inertial.inertias = np.zeros(6)

            # Spawn
            sdf = ModelSDF(
                name="water",
                pose=np.zeros(6),
                links=[link],
                joints=[],
                units=self.units
            )
            sdf.write(filename="water.sdf")
            print(os.getcwd() + "/water.sdf")
            self._identity = pybullet.loadSDF(
                os.getcwd() + "/water.sdf",
                useMaximalCoordinates=0,
                globalScaling=1
            )[0]

        # Dynamics properties
        group = 0  #other objects don't collide with me
        mask = 0  # don't collide with any other object
        pybullet. setCollisionFilterGroupMask(
            self.identity,
            -1,
            group,
            mask
        )


class ArenaWater(ArenaRamp):
    """ArenaRamp"""

    def __init__(self, units, ramp_angle=None, elements=None):
        water_surface = -0.1
        super(ArenaWater, self).__init__(
            units=units,
            ramp_angle=ramp_angle,
            elements=[Water(units, water_surface)]
        )
        self.water_surface = water_surface

    @property
    def water(self):
        """Floor"""
        return self.elements[1]
