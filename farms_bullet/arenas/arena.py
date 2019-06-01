"""Arena"""

from .create import create_scene
from ..simulations.element import SimulationElement
import numpy as np
import pybullet as p


class Floor(SimulationElement):
    """Floor"""

    def __init__(self, position):
        super(Floor, self).__init__()
        self._position = position

    def spawn(self):
        """Spawn floor"""
        self._identity = self.from_urdf(
            "plane.urdf",
            basePosition=self._position
        )


class Arena:
    """Documentation for Arena"""

    def __init__(self, elements):
        super(Arena, self).__init__()
        self.elements = elements

    def spawn(self):
        """Spawn"""
        for element in self.elements:
            element.spawn()


class FlooredArena(Arena):
    """Arena with floor"""

    def __init__(self, position=None):
        super(FlooredArena, self).__init__(
            [Floor(position if position is not None else [0, 0, -0.1])]
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

class ArenaExperiment1:
    def __init__(self, rampAngle=20,  rampHalfLength=4, rampHalfWidth=2.5, rampHalfHeight=0.1,
                 upperPlatLength=4, upperPlatWidth=2.5,
                 lowerPlatLength=4, lowerPlatWidth=2.5,
                 barrierHeight=0.0, barrierWidth=0.1, barrierThickness=0.2,
                 lowerPlat_x = 0, lowerPlat_y = 0, lowerPlat_z = -0.2,
                 creatBarrier = bool(True)):
        """creation of the salamander arena with a simple ramp and two
        platform in order to evolve the salamander
        datum : 6 march 2019"""

        self.rampAngle = np.deg2rad(rampAngle)
        self.rampHalfLength = rampHalfLength
        self.rampHalfWidth = rampHalfWidth
        self.rampHalfHeight = rampHalfHeight

        #upper platform attributes
        self.upperPlatLength = upperPlatLength
        self.upperPlatWidth = upperPlatWidth

        #lower platform attributes
        self.lowerPlatLength = lowerPlatLength
        self.lowerPlatWidth = lowerPlatWidth
        self.lowerPlat_x = lowerPlat_x
        self.lowerPlat_y = lowerPlat_y
        self.lowerPlat_z = lowerPlat_z

        #option to spawn
        self.createBarrier = creatBarrier
        self.barrierHeight = barrierHeight
        self.barrierWidth = barrierWidth
        self.barrierThickness = barrierThickness

    def spawn(self, moving_ramp = bool(False), showGround=bool(True)):
        """create the arena with the ramp"""
        if showGround:
            p.createCollisionShape(p.GEOM_PLANE)
            p.createMultiBody(0, 0)

        # defining the dimension of the ramp
        ramp_dimensions = [self.rampHalfLength, self.rampHalfWidth, self.rampHalfHeight]
        upper_platform_dimensions = [self.upperPlatLength, self.upperPlatWidth, self.rampHalfHeight]
        lower_platform_dimensions = [self.lowerPlatLength, self.lowerPlatWidth, self.rampHalfHeight]

        # creating the collision shape for the 3 geometries
        upperPlatId = p.createCollisionShape(p.GEOM_BOX,
                                             halfExtents=upper_platform_dimensions,
                                             collisionFramePosition=[0, self.upperPlatWidth, 0])
        lowerPlatId = p.createCollisionShape(p.GEOM_BOX,
                                             halfExtents=lower_platform_dimensions,
                                             collisionFramePosition=[0, self.lowerPlatWidth, 0])
        rampBoxId = p.createCollisionShape(p.GEOM_BOX,
                                           halfExtents=ramp_dimensions,
                                           collisionFramePosition=[0, self.rampHalfWidth, 0])

        #creating the side barrier collision shape
        LowerSideBarrier_dimensions = [self.barrierWidth, self.lowerPlatWidth, self.barrierHeight]
        lowerSideBarrierId = p.createCollisionShape(p.GEOM_BOX, halfExtents=LowerSideBarrier_dimensions)
        lowerSideBarrierVisId = p.createVisualShape(p.GEOM_BOX, halfExtents=LowerSideBarrier_dimensions,
                                                    rgbaColor=[0,0,0,1])
        barrier_mass = 0

        #creating the front barrier collision shpae
        frontBarrier_dimensions = [self.lowerPlatLength, self.barrierWidth, self.barrierHeight]
        frontBarrierId = p.createCollisionShape(p.GEOM_BOX, halfExtents=frontBarrier_dimensions)
        frontBarrierVisId = p.createVisualShape(p.GEOM_BOX, halfExtents=frontBarrier_dimensions,
                                                rgbaColor=[0,0,0,1])


        #creating ramp side barrier
        RampSideBarrier_dimensions = [self.barrierWidth, self.rampHalfWidth, self.barrierHeight]
        RampSideBarrierId = p.createCollisionShape(p.GEOM_BOX, halfExtents=RampSideBarrier_dimensions)
        RampSideBarrierVisId = p.createVisualShape(p.GEOM_BOX, halfExtents=RampSideBarrier_dimensions,
                                                   rgbaColor=[0,0,0,1])


        #creating upper side barrier
        UpperSideBarrier_dimensions = [self.barrierWidth, self.upperPlatWidth, self.barrierHeight]
        UpperSideBarrierId = p.createCollisionShape(p.GEOM_BOX, halfExtents=UpperSideBarrier_dimensions)
        UpperSideBarrierVisId = p.createVisualShape(p.GEOM_BOX, halfExtents=UpperSideBarrier_dimensions,
                                                    rgbaColor=[0,0,0,1])


        ramp_orientation = p.getQuaternionFromEuler([self.rampAngle, 0, 0])
        upper_orientation = p.getQuaternionFromEuler([-self.rampAngle, 0, 0])

        #computing the visual identity for the 3 links
        rampVisId = p.createVisualShape(p.GEOM_BOX,
                                            halfExtents=ramp_dimensions,
                                            visualFramePosition=[0, self.rampHalfWidth, 0],
                                            rgbaColor=[0.8, 0.8, 0.8])
        lowerPlatVisId = p.createVisualShape(p.GEOM_BOX,
                                                 halfExtents=lower_platform_dimensions,
                                                 rgbaColor=[0.8, 0.8, 0.8])
        upperPlatVisId = p.createVisualShape(p.GEOM_BOX,
                                                 halfExtents=upper_platform_dimensions,
                                                 visualFramePosition=[0, self.upperPlatWidth, 0],
                                                 rgbaColor=[0.8, 0.8, 0.8])



        #defining the initial position of the ground
        lowerPlatPosition = [self.lowerPlat_x, self.lowerPlat_y, self.lowerPlat_z]
        lowerPlatOrientation = [0, 0, 1]
        if moving_ramp==True:
            mass_ramp = 1
            mass_upper = 1
        else:
            mass_ramp = 0
            mass_upper = 0

        linkCollisionShapeIndices_barrier = [lowerSideBarrierId, lowerSideBarrierId, frontBarrierId,
                                             RampSideBarrierId, RampSideBarrierId, UpperSideBarrierId,
                                             UpperSideBarrierId]

        linkVisualShapeIndices_barrier = [lowerSideBarrierVisId, lowerSideBarrierVisId,
                                          frontBarrierVisId, RampSideBarrierVisId, RampSideBarrierVisId,
                                          UpperSideBarrierVisId, UpperSideBarrierVisId]

        linkPositions_barrier = [[self.lowerPlatLength, 0, self.barrierHeight],
                                 [-self.lowerPlatLength, 0, self.barrierHeight],
                                 [0, -self.lowerPlatWidth, self.barrierHeight],
                                 [-self.rampHalfLength, self.rampHalfWidth, self.barrierHeight],
                                 [self.rampHalfLength, self.rampHalfWidth, self.barrierHeight],
                                 [self.upperPlatLength, self.upperPlatWidth, self.barrierHeight],
                                 [-self.upperPlatLength, self.upperPlatWidth, self.barrierHeight]]

        linkInertialFramePositions_barrier = [[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],
                                                 [0, 0, 0],[0, 0, 0],[0, 0, 0]]
        linkInertialFrameOrientations_barrier = [[0, 0, 0, 1],[0, 0, 0, 1],[0, 0, 0, 1],
                                         [0, 0, 0, 1],[0, 0, 0, 1],[0, 0, 0, 1],[0, 0, 0, 1]]
        ParentIndices_barrier = [0, 0, 0, 1, 1, 2, 2]

        jointTypes_barrier = [p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_REVOLUTE,
                      p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_REVOLUTE]

        link_Masses = [mass_ramp, mass_upper]
        linkCollisionShapeIndices = [rampBoxId, upperPlatId]
        linkOrientations = [ramp_orientation, upper_orientation]
        linkVisualShapeIndices = [rampVisId, upperPlatVisId]
        linkPositions = [[0, self.lowerPlatWidth, 0],
                         [0, self.lowerPlatWidth + self.rampHalfWidth, 0]]
        linkInertialFramePositions = [[0, 0, 0], [0, 0, 0]]
        linkInertialFrameOrientations = [[0, 0, 0, 1], [0, 0, 0, 1]]
        ParentIndices = [0, 1]
        jointTypes = [p.JOINT_REVOLUTE, p.JOINT_REVOLUTE]
        axis = [[1, 0, 0], [1, 0, 0]]

        if self.createBarrier:
            for i in np.arange(0,7):
                linkOrientations.append([0, 0, 0, 1])
                link_Masses.append(barrier_mass)
                linkCollisionShapeIndices.append(linkCollisionShapeIndices_barrier[i])
                linkVisualShapeIndices.append(linkVisualShapeIndices_barrier[i])
                linkPositions.append(linkPositions_barrier[i])
                linkInertialFramePositions.append(linkInertialFramePositions_barrier[i])
                linkInertialFrameOrientations.append(linkInertialFrameOrientations_barrier[i])
                ParentIndices.append(ParentIndices_barrier[i])
                jointTypes.append(jointTypes_barrier[i])
                axis.append([1,0,0])

        arenaId = p.createMultiBody(baseMass=0,
                                      baseCollisionShapeIndex=lowerPlatId,
                                      baseVisualShapeIndex=lowerPlatVisId,
                                      basePosition=lowerPlatPosition,
                                      baseOrientation=lowerPlatOrientation,
                                      linkMasses=link_Masses,
                                      linkCollisionShapeIndices=linkCollisionShapeIndices,
                                      linkVisualShapeIndices=linkVisualShapeIndices,
                                      linkPositions=linkPositions,
                                      linkOrientations=linkOrientations,
                                      linkInertialFramePositions=linkInertialFramePositions,
                                      linkInertialFrameOrientations=linkInertialFrameOrientations,
                                      linkParentIndices=ParentIndices,
                                      linkJointTypes=jointTypes,
                                      linkJointAxis=axis)

