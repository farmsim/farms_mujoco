"""Simon's animat"""

import numpy as np
import pybullet
from ...animats.animat import Animat
from ...animats.link import AnimatLink
from .animat_data import SimonData


class SimonAnimat(Animat):
    """Documentation for SimonAnimat"""

    def __init__(self, options, timestep, n_iterations):
        super(SimonAnimat, self).__init__(None, options)
        self.timestep = timestep
        self.n_iterations = n_iterations
        self.sensors = None
        self.data = SimonData.from_options(
            state=None,
            options=self.options,
            n_iterations=n_iterations
        )

    def spawn(self):
        """Spawn"""
        print("Spawning animat")
        base_link = AnimatLink(
            size=[0.1, 0.05, 0.02],
            geometry=pybullet.GEOM_BOX,
            position=[0, 0, 0],
            orientation=[0, 0, 0],
            f_position=[0, 0, 0],
            mass=2
        )
        # Upper legs
        upper_legs_positions = np.array([
            [0.1, 0.08, 0.01],
            [0.1, -0.08, 0.01],
            [-0.1, 0.08, 0.01],
            [-0.1, -0.08, 0.01]
        ])
        upper_legs = [
            AnimatLink(
                size=[0.02, 0.02, 0.02],
                geometry=pybullet.GEOM_BOX,
                position=position,
                orientation=[0, 0, 0],
                f_position=position,
                f_orientation=[0, 0, 0],
                frame_position=[0, 0, -0.02],
                joint_axis=[1, 0, 0],
                mass=0.5
            ) for position in upper_legs_positions
        ]
        upper_legs[0].parent = 0
        upper_legs[1].parent = 0
        upper_legs[2].parent = 0
        upper_legs[3].parent = 0
        # Lower legs
        lower_legs = [
            AnimatLink(
                # size=[0.02, 0.02, 0.04],
                geometry=pybullet.GEOM_CAPSULE,
                radius=0.02,
                height=0.02,
                position=[0, 0, -0.03],
                orientation=[0, 0, 0],
                f_position=[0, 0, -0.03],
                f_orientation=[0, 0, 0],
                frame_position=[0, 0, -0.03],
                joint_axis=[0, 1, 0],
                mass=0.5
            ) for upper_pos in upper_legs_positions
        ]
        lower_legs[0].parent = 1
        lower_legs[1].parent = 2
        lower_legs[2].parent = 3
        lower_legs[3].parent = 4
        links = upper_legs + lower_legs
        # Create body
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
        # Joints dynamics
        n_joints = pybullet.getNumJoints(self.identity)
        for joint in range(n_joints):
            pybullet.changeDynamics(
                self.identity,
                joint,
                lateralFriction=0.1,
                spinningFriction=0,
                rollingFriction=0,
                linearDamping=0,
                jointDamping=0
            )
        print("Number of joints: {}".format(n_joints))
        for joint in range(n_joints):
            pybullet.resetJointState(
                self.identity,
                joint,
                targetValue=0,
                targetVelocity=0
            )
        # pybullet.setJointMotorControlArray(
        #     self.identity,
        #     np.arange(n_joints),
        #     pybullet.POSITION_CONTROL,
        #     targetPositions=np.ones(n_joints),
        #     forces=1e3*np.ones(n_joints)
        # )
        # pybullet.setJointMotorControlArray(
        #     self.identity,
        #     np.arange(n_joints),
        #     pybullet.TORQUE_CONTROL,
        #     # targetPositions=np.ones(n_joints),
        #     forces=1e1*np.ones(n_joints)
        # )
        # Cancel controller
        pybullet.setJointMotorControlArray(
            self.identity,
            np.arange(n_joints),
            pybullet.VELOCITY_CONTROL,
            forces=np.zeros(n_joints)
        )

        pybullet.setGravity(0,0,-9.81)
        # pybullet.setRealTimeSimulation(1)

        pybullet.getNumJoints(self.identity)
        for i in range(pybullet.getNumJoints(self.identity)):
            print("Joint information: {}".format(
                pybullet.getJointInfo(self.identity, i)
            ))
