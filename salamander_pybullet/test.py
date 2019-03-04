"""Salamander simulation with pybullet"""

import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

import pybullet_data
import pybullet

import casadi as cas


def parse_args():
    """ Parse arguments """
    parser = argparse.ArgumentParser(description='Salamander simulation')
    parser.add_argument(
        '-f', '--free_camera',
        action='store_true',
        dest='free_camera',
        default=False,
        help='Allow for free camera (User controlled)'
    )
    parser.add_argument(
        '-r', '--rotating_camera',
        action='store_true',
        dest='rotating_camera',
        default=False,
        help='Enable rotating camera'
    )
    parser.add_argument(
        '-t', '--top_camera',
        action='store_true',
        dest='top_camera',
        default=False,
        help='Enable top view camera'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        dest='fast',
        default=False,
        help='Remove real-time limiter'
    )
    parser.add_argument(
        '--record',
        action='store_true',
        dest='record',
        default=False,
        help='Record video'
    )
    return parser.parse_args()


class SineControl:
    """SineControl"""

    def __init__(self, amplitude, frequency, phase, offset):
        super(SineControl, self).__init__()
        self.amplitude = amplitude
        self._angular_frequency = 2*np.pi*frequency
        self.phase = phase
        self.offset = offset

    @property
    def angular_frequency(self):
        """Angular frequency"""
        return self._angular_frequency

    @angular_frequency.setter
    def angular_frequency(self, value):
        self._angular_frequency = value

    def position(self, phase):
        """"Position"""
        return self.amplitude*np.sin(
            phase + self.phase
        ) + self.offset

    def velocity(self, phase):
        """Velocity"""
        return self._angular_frequency*self.amplitude*np.cos(
            phase + self.phase
        )


class ControlPDF(dict):
    """ControlPDF"""

    def __init__(self, p=1, d=0, f=0):
        super(ControlPDF, self).__init__()
        self["p"] = p
        self["d"] = d
        self["f"] = f

    @property
    def p_term(self):
        """Proportfonal term"""
        return self["p"]

    @property
    def d_term(self):
        """Derivative term"""
        return self["d"]

    @property
    def f_term(self):
        """Max force term"""
        return self["f"]


class JointController:
    """JointController"""

    def __init__(self, joint, sine, pdf, **kwargs):
        super(JointController, self).__init__()
        self._joint = joint
        self._sine = sine
        self._pdf = pdf
        self._is_body = kwargs.pop("is_body", False)

    def cmds(self, phase):
        """Commands"""
        return {
            "pos": self._sine.position(phase),
            "vel": self._sine.velocity(phase)
        }

    def update(self, phase):
        """Update"""
        return {
            "joint": self._joint,
            "cmd": self.cmds(phase),
            "pdf": self._pdf
        }

    def angular_frequency(self):
        """Angular frequency"""
        return self._sine.angular_frequency

    def set_frequency(self, frequency):
        """Set frequency"""
        self._sine.angular_frequency = 2*np.pi*frequency

    def set_body_offset(self, body_offset):
        """Set body offset"""
        if self._is_body:
            self._sine.offset = body_offset


class Network:
    """Controller network"""

    def __init__(self, controllers, **kwargs):
        super(Network, self).__init__()
        size = len(controllers)
        freqs = cas.MX.sym('freqs', size)
        ode = {
            "x": cas.MX.sym('x', size),
            "p": freqs,
            "ode": freqs
        }

        # Construct a Function that integrates over 4s
        self.ode = cas.integrator(
            'oscillator',
            'cvodes',
            ode,
            {
                "t0": 0,
                "tf": kwargs.pop("timestep", 1e-3),
                "jit": True,
                "step0": 1e-3,
                "abstol": 1e-3,
                "reltol": 1e-3
            },
        )
        self.phases = np.zeros(size)

    def control_step(self, freqs):
        """Control step"""
        self.phases = np.array(
            self.ode(
                x0=self.phases,
                p=freqs
            )["xf"][:, 0]
        )
        return self.phases


class RobotController:
    """RobotController"""

    def __init__(self, robot, joints_controllers, timestep=1e-3):
        super(RobotController, self).__init__()
        self.robot = robot
        self.controllers = joints_controllers
        self.network = Network(self.controllers, timestep=timestep)

    def control(self, verbose=False):
        """Control"""
        phases = self.network.control_step([
            float(controller.angular_frequency())
            for controller in self.controllers
        ])
        if verbose:
            tic = time.time()
        controls = [
            controller.update(phases[i])
            for i, controller in enumerate(self.controllers)
        ]
        if verbose:
            toc = time.time()
            print("Time to copy phases: {} [s]".format(toc-tic))
        pybullet.setJointMotorControlArray(
            self.robot,
            [ctrl["joint"] for ctrl in controls],
            pybullet.POSITION_CONTROL,
            targetPositions=[ctrl["cmd"]["pos"] for ctrl in controls],
            targetVelocities=[ctrl["cmd"]["vel"] for ctrl in controls],
            positionGains=[ctrl["pdf"]["p"] for ctrl in controls],
            velocityGains=[ctrl["pdf"]["d"] for ctrl in controls],
            forces=[ctrl["pdf"]["f"] for ctrl in controls]
        )

    def update_frequency(self, frequency):
        """Update frequency"""
        for controller in self.controllers:
            controller.set_frequency(frequency)

    def update_body_offset(self, body_offset):
        """Update body offset"""
        for controller in self.controllers:
            controller.set_body_offset(body_offset)


class SalamanderController(RobotController):
    """RobotController"""

    @classmethod
    def gait(cls, robot, joints, gait, **kwargs):
        """Salamander gait controller"""
        return (
            cls.walking(robot, joints, **kwargs)
            if gait == "walking" else
            cls.swimming(robot, joints, **kwargs)
            if gait == "swimming" else
            cls.standing(robot, joints, **kwargs)
        )

    @classmethod
    def standing(cls, robot, joints, **kwargs):
        """Salamander standing controller"""
        n_body_joints = kwargs.pop("n_body_joints", 11)
        joint_controllers_body = [
            JointController(
                joint=joints["joint_link_body_{}".format(joint_i+1)],
                sine=SineControl(
                    amplitude=0,
                    frequency=0,
                    phase=0,
                    offset=0
                ),
                pdf=(
                    ControlPDF(p=1e-1, d=1e0, f=1e1)
                ),
                is_body=True
            )
            for joint_i in range(n_body_joints)
        ]
        joint_controllers_legs = [
            JointController(
                joint=joints["joint_link_leg_{}_{}_{}".format(
                    leg_i,
                    side,
                    joint_i
                )],
                sine=SineControl(
                    amplitude=0.0,
                    frequency=0,
                    phase=0,
                    offset=(
                        0 if joint_i == 0
                        else np.pi/16 if joint_i == 1
                        else np.pi/8
                    )
                ),
                pdf=ControlPDF(p=1e-1, d=1e0, f=1e1)
            )
            for leg_i in range(2)
            for side_i, side in enumerate(["L", "R"])
            for joint_i in range(3)
        ]
        return cls(
            robot,
            joint_controllers_body + joint_controllers_legs,
            timestep=kwargs.pop("timestep", 1e-3)
        )

    @classmethod
    def walking(cls, robot, joints, **kwargs):
        """Salamander walking controller"""
        n_body_joints = kwargs.pop("n_body_joints", 11)
        frequency = kwargs.pop("frequency", 1)
        joint_controllers_body = [
            JointController(
                joint=joints["joint_link_body_{}".format(joint_i+1)],
                sine=SineControl(
                    amplitude=(
                        0.2*np.sin(2*np.pi*joint_i/n_body_joints - np.pi/4)
                    ),
                    frequency=frequency,
                    phase=0,
                    offset=0
                ),
                pdf=(
                    ControlPDF(p=1e-1, d=1e0, f=1e1)
                ),
                is_body=True
            )
            for joint_i in range(n_body_joints)
        ]
        joint_controllers_legs = [
            JointController(
                joint=joints["joint_link_leg_{}_{}_{}".format(
                    leg_i,
                    side,
                    joint_i
                )],
                sine=SineControl(
                    amplitude=(
                        0.8
                        if joint_i == 0
                        else np.pi/16 if joint_i == 1
                        else np.pi/8
                    ),
                    frequency=frequency,
                    phase=(
                        - np.pi*np.abs(leg_i-side_i)
                        - (
                            0 if joint_i == 0
                            else 0.5*np.pi
                        )
                        + 0*float(  # Turning
                            (0.5)*np.pi*np.sign(np.abs(leg_i-side_i) - 0.5)
                            if joint_i == 2
                            else 0
                        )
                    ),
                    offset=(
                        0 if joint_i == 0
                        else np.pi/16 if joint_i == 1
                        else np.pi/8
                    )
                ),
                pdf=ControlPDF(p=1e-1, d=1e0, f=1e1)
            )
            for leg_i in range(2)
            for side_i, side in enumerate(["L", "R"])
            for joint_i in range(3)
        ]
        return cls(
            robot,
            joint_controllers_body + joint_controllers_legs,
            timestep=kwargs.pop("timestep", 1e-3)
        )

    @classmethod
    def swimming(cls, robot, joints, **kwargs):
        """Salamander swimming controller"""
        n_body_joints = kwargs.pop("n_body_joints", 11)
        frequency = kwargs.pop("frequency", 1)
        joint_controllers_body = [
            JointController(
                joint=joints["joint_link_body_{}".format(joint_i+1)],
                sine=SineControl(
                    amplitude=0.1+joint_i*0.4/n_body_joints,
                    frequency=frequency,
                    phase=-2*np.pi*joint_i/11,
                    offset=0
                ),
                pdf=(
                    ControlPDF(p=1e-1, d=1e0, f=1e1)
                ),
                is_body=True
            )
            for joint_i in range(n_body_joints)
        ]
        joint_controllers_legs = [
            JointController(
                joint=joints["joint_link_leg_{}_{}_{}".format(
                    leg_i,
                    side,
                    joint_i
                )],
                sine=SineControl(
                    amplitude=0.0,
                    frequency=0,
                    phase=0,
                    offset=-2*np.pi/5 if joint_i == 0 else 0
                ),
                pdf=ControlPDF(p=1e-1, d=1e0, f=1e1)
            )
            for leg_i in range(2)
            for side_i, side in enumerate(["L", "R"])
            for joint_i in range(3)
        ]
        return cls(
            robot,
            joint_controllers_body + joint_controllers_legs,
            timestep=kwargs.pop("timestep", 1e-3)
        )


def init_engine():
    """Initialise engine"""
    print(pybullet.getAPIVersion())
    pybullet.connect(pybullet.GUI, options="--minGraphicsUpdateTimeMs=32000")
    pybullet_path = pybullet_data.getDataPath()
    print("Adding pybullet data path {}".format(pybullet_path))
    pybullet.setAdditionalSearchPath(pybullet_path)


def init_physics(timestep, gait="walking"):
    """Initialise physics"""
    pybullet.resetSimulation()
    pybullet.setGravity(0, 0, -1e-2 if gait == "swimming" else -9.81)
    pybullet.setTimeStep(timestep)
    pybullet.setRealTimeSimulation(0)
    pybullet.setPhysicsEngineParameter(
        fixedTimeStep=timestep,
        numSolverIterations=50
    )
    print("Physics parameters:\n{}".format(
        pybullet.getPhysicsEngineParameters()
    ))


def spawn_models():
    """Spawn models"""
    robot = pybullet.loadSDF(
        "/home/jonathan/.gazebo/models/biorob_salamander/model.sdf",
        # useMaximalCoordinates=1
    )[0]
    # robot = pybullet.loadSDF(
    #     "/home/jonathan/.gazebo/models/biorob_centipede/model.sdf"
    # )[0]
    plane = pybullet.loadURDF(
        "plane.urdf",
        basePosition=[0, 0, -0.1]
    )
    return robot, plane


def get_joints(robot):
    """Get joints"""
    print("Robot: {}".format(robot))
    n_joints = pybullet.getNumJoints(robot)
    print("Number of joints: {}".format(n_joints))

    joint_index = 2
    joint_info = pybullet.getJointInfo(robot, joint_index)
    print(joint_info)

    # joint_positions = {
    #     name.decode("UTF-8"): state[0]
    #     for name, state in zip(
    #         [pybullet.getJointInfo(robot, j)[1] for j in range(n_joints)],
    #         pybullet.getJointStates(robot, range(n_joints)),
    #     )
    # }
    links = {
        info[12].decode("UTF-8"): info[16] + 1
        for info in [
            pybullet.getJointInfo(robot, j)
            for j in range(n_joints)
        ]
    }
    joints = {
        info[1].decode("UTF-8"): info[0]
        for info in [
            pybullet.getJointInfo(robot, j)
            for j in range(n_joints)
        ]
    }
    return links, joints, n_joints


def camera_view(robot, target_pos=None, **kwargs):
    """Camera view"""
    camera_filter = kwargs.pop("camera_filter", 1e-3)
    yaw_speed = kwargs.pop("yaw_speed", 0)
    camInfo = pybullet.getDebugVisualizerCamera()
    # curTargetPos = camInfo[11]
    # distance=camInfo[10]
    # yaw = camInfo[8]
    # pitch=camInfo[9]
    # targetPos = [0.95*curTargetPos[0]+0.05*humanPos[0],0.95*curTargetPos[1]+0.05*humanPos[1],curTargetPos[2]]
    timestep = kwargs.pop("timestep", 1e-3)
    pitch = kwargs.pop("pitch", camInfo[9])
    yaw = kwargs.pop("yaw", camInfo[8]) + yaw_speed*timestep
    distance = kwargs.pop("distance", camInfo[10])
    # sim_time*360/10 if clargs.rotating_camera else 0
    # yaw = kwargs.pop("yaw", 0)
    target_pos = (
        (
            (1-camera_filter)*target_pos
            + camera_filter*np.array(
                pybullet.getBasePositionAndOrientation(robot)[0]
            )
        )
        if target_pos is not None
        else np.array(
            pybullet.getBasePositionAndOrientation(robot)[0]
        )
    )
    pybullet.resetDebugVisualizerCamera(
        cameraDistance=distance,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=target_pos
    )
    return target_pos


def viscous_swimming(robot, links):
    """Viscous swimming"""
    # Swimming
    forces_torques = np.zeros([2, 10, 3])
    for link_i in range(1, 11):
        link_state = pybullet.getLinkState(
            robot,
            links["link_body_{}".format(link_i)],
            computeLinkVelocity=1,
            computeForwardKinematics=1
        )
        link_orientation_inv = np.linalg.inv(np.array(
            pybullet.getMatrixFromQuaternion(link_state[5])
        ).reshape([3, 3]))
        link_velocity = np.dot(link_orientation_inv, link_state[6])
        link_angular_velocity = np.dot(link_orientation_inv, link_state[7])
        forces_torques[0, link_i-1, :] = (
            np.array([-1e-1, -1e0, -1e0])*link_velocity
        )
        pybullet.applyExternalForce(
            robot,
            links["link_body_{}".format(link_i)],
            forceObj=forces_torques[0, link_i-1, :],
            posObj=[0, 0, 0],
            flags=pybullet.LINK_FRAME
        )
        forces_torques[1, link_i-1, :] = (
            np.array([-1e-2, -1e-2, -1e-2])*link_angular_velocity
        )
        pybullet.applyExternalTorque(
            robot,
            links["link_body_{}".format(link_i+1)],
            torqueObj=forces_torques[1, link_i-1, :],
            flags=pybullet.LINK_FRAME
        )
    return forces_torques


def record_camera(position, yaw, pitch, distance):
    """Record camera"""
    return pybullet.getCameraImage(
        width=640,
        height=480,
        viewMatrix=pybullet.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=position,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=0,
            upAxisIndex=2
        ),
        projectionMatrix = pybullet.computeProjectionMatrixFOV(
            fov=60,
            aspect=600/360,
            nearVal=0.1,
            farVal=5
        ),
        renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
        flags=pybullet.ER_NO_SEGMENTATION_MASK
    )[2]


def init_simulation(timestep, gait="walking"):
    """Initialise simulation"""
    # Physics
    init_physics(timestep, gait)

    # Spawn models
    robot, plane = spawn_models()

    # Links and joints
    links, joints, _ = get_joints(robot)
    print("Links ids:\n{}".format(
        "\n".join([
            "  {}: {}".format(
                name,
                links[name]
            )
            for name in links
        ])
    ))
    print("Joints ids:\n{}".format(
        "\n".join([
            "  {}: {}".format(
                name,
                joints[name]
            )
            for name in joints
        ])
    ))
    return robot, links, joints, plane


def user_parameters(gait, frequency):
    """User parameters"""
    play_id = pybullet.addUserDebugParameter(
        paramName="Play",
        rangeMin=0,
        rangeMax=1,
        startValue=1
    )
    rtl_id = pybullet.addUserDebugParameter(
        paramName="Real-time limiter",
        rangeMin=1e-3,
        rangeMax=3,
        startValue=1
    )
    gait_id = pybullet.addUserDebugParameter(
        paramName="Gait",
        rangeMin=0,
        rangeMax=2,
        startValue=(
            0 if gait == "standing"
            else 1 if gait == "walking"
            else 2
        )
    )
    freq_id = pybullet.addUserDebugParameter(
        paramName="Frequency",
        rangeMin=0,
        rangeMax=5,
        startValue=frequency
    )
    body_offset_id = pybullet.addUserDebugParameter(
        paramName="Body offset",
        rangeMin=-np.pi/8,
        rangeMax=np.pi/8,
        startValue=0
    )
    for part in ["body", "legs"]:
        for pdf in ["p", "d", "f"]:
            pybullet.addUserDebugParameter(
                paramName="{}_{}".format(part, pdf),
                rangeMin=0,
                rangeMax=10,
                startValue=0.1
            )
    return play_id, rtl_id, gait_id, freq_id, body_offset_id


def test_debug_info():
    """Test debug info"""
    pybullet.addUserDebugLine(
        lineFromXYZ=[0, 0, -0.09],
        lineToXYZ=[-3, 0, -0.09],
        lineColorRGB=[0.1, 0.5, 0.9],
        lineWidth=10,
        lifeTime=0
    )
    text = pybullet.addUserDebugText(
        text="BIOROB",
        textPosition=[-3, 0.1, -0.09],
        textColorRGB=[0, 0, 0],
        textSize=1,
        lifeTime=0,
        textOrientation=[0, 0, 0, 1],
        # parentObjectUniqueId
        # parentLinkIndex
        # replaceItemUniqueId
    )


def real_time_handing(timestep, tic_rt, toc_rt, rtl=1.0, **kwargs):
    """Real-time handling"""
    sleep_rtl = timestep/rtl - (toc_rt - tic_rt)
    rtf = timestep / (toc_rt - tic_rt)
    tic = time.time()
    sleep_rtl = np.clip(sleep_rtl, a_min=0, a_max=1)
    if sleep_rtl > 0:
        while time.time() - tic < sleep_rtl:
            time.sleep(0.1*sleep_rtl)
    if rtf < 1:
        print("Slower than real-time: {} %".format(100*rtf))
        time_plugin = kwargs.pop("time_plugin", False)
        time_control = kwargs.pop("time_control", False)
        time_sim = kwargs.pop("time_sim", False)
        if time_plugin:
            print("  Time in py_plugins: {} [ms]".format(time_plugin))
        if time_control:
            print("    Time in control: {} [ms]".format(time_control))
        if time_sim:
            print("  Time in simulation: {} [ms]".format(time_sim))


def create_scene(plane):
    """Create scene"""

    # pybullet.createCollisionShape(pybullet.GEOM_PLANE)
    pybullet.createMultiBody(0,0)

    sphereRadius = 0.01
    colSphereId = pybullet.createCollisionShape(
        pybullet.GEOM_SPHERE,
        radius=sphereRadius
    )
    colCylinderId = pybullet.createCollisionShape(
        pybullet.GEOM_CYLINDER,
        radius=sphereRadius,
        height=1
    )
    colBoxId = pybullet.createCollisionShape(
        pybullet.GEOM_BOX,
        halfExtents=[sphereRadius, sphereRadius, sphereRadius]
    )

    mass = 1
    visualShapeId = -1


    link_Masses=[1]
    linkCollisionShapeIndices=[colBoxId]
    linkVisualShapeIndices=[-1]
    linkPositions=[[0,0,0.11]]
    linkOrientations=[[0,0,0,1]]
    linkInertialFramePositions=[[0,0,0]]
    linkInertialFrameOrientations=[[0,0,0,1]]
    indices=[0]
    jointTypes=[pybullet.JOINT_REVOLUTE]
    axis=[[0,0,1]]

    j = 0
    k = 0
    for i in range (30):
        for j in range (10):
            basePosition = [
                -3 - i*10*sphereRadius,
                -0.5 + j*10*sphereRadius,
                sphereRadius/2
            ]
            baseOrientation = [0, 0, 0, 1]
            sphereUid = pybullet.createMultiBody(
                mass,
                colCylinderId,
                visualShapeId,
                basePosition,
                baseOrientation
            )
            cid = pybullet.createConstraint(
                sphereUid, -1,
                plane, -1,
                pybullet.JOINT_FIXED,
                [0, 0, 1],
                [0, 0, 0],
                basePosition
            )

            pybullet.changeDynamics(
                sphereUid, -1,
                spinningFriction=0.001,
                rollingFriction=0.001,
                linearDamping=0.0
            )


def get_links_contacts(robot, links, ground):
    """Contacts"""
    contacts = [
        pybullet.getContactPoints(robot, ground, link, -1)
        for link in links
    ]
    forces = [
        np.sum([contact[9] for contact in contacts[link_i]])
        if contacts
        else 0
        for link_i, _ in enumerate(links)
    ]
    return contacts, forces


def get_joints_force_torque(robot, joints):
    """Force-torque on joints"""
    return [
        pybullet.getJointState(robot, joint)[2]
        for joint in joints
    ]


def get_joints_commands(robot, joints):
    """Force-torque on joints"""
    return [
        pybullet.getJointState(robot, joint)[3]
        for joint in joints
    ]


def main(clargs):
    """Main"""
    # Initialise engine
    init_engine()

    # Parameters
    # gait = "standing"
    gait = "walking"
    # gait = "swimming"
    timestep = 1e-3

    # Initialise
    robot, links, joints, plane = init_simulation(timestep, gait)

    # Apply motor damping
    for j in range (pybullet.getNumJoints(robot)):
        pybullet.changeDynamics(robot, j, linearDamping=0, angularDamping=1e-2)

    # Create scene
    add_obstacles = False
    if add_obstacles:
        create_scene(plane)

    # Controller
    frequency = 1 if gait == "walking" else 2
    body_offset = 0
    controller = SalamanderController.gait(
        robot,
        joints,
        gait=gait,
        frequency=frequency,
        timestep=timestep
    )

    # Camera
    camera_pitch = -89 if clargs.top_camera else -45
    target_pos = camera_view(robot, yaw=0, pitch=camera_pitch, distance=1)

    # User parameters
    user_params = user_parameters(gait, frequency)
    play_id, rtl_id, gait_id, freq_id, body_offset_id = user_params

    # Debug info
    test_debug_info()

    # Simulation time
    tic = time.time()
    tot_sim_time = 0
    times = np.arange(0, 10, timestep)
    forces_torques = np.zeros([len(times), 2, 10, 3])
    sim_step = 0

    # Contact sensors
    contact_forces = np.zeros([len(times), 4])
    feet = [
        "link_leg_0_L_3",
        "link_leg_0_R_3",
        "link_leg_1_L_3",
        "link_leg_1_R_3"
    ]

    # Force-torque sensors
    feet_ft = np.zeros([len(times), 4, 6])
    joints_sensors = [
        "joint_link_leg_0_L_3",
        "joint_link_leg_0_R_3",
        "joint_link_leg_1_L_3",
        "joint_link_leg_1_R_3"
    ]
    for joint in joints_sensors:
        pybullet.enableJointForceTorqueSensor(robot, joints[joint])

    # Commands
    joints_commanded_body = [
        "joint_link_body_{}".format(joint_i+1)
        for joint_i in range(11)
    ]
    joints_commanded_legs = [
        "joint_link_leg_{}_{}_{}".format(leg_i, side, joint_i)
        for leg_i in range(2)
        for side in ["L", "R"]
        for joint_i in range(3)
    ]
    joints_cmds_body = np.zeros([len(times), len(joints_commanded_body)])
    joints_cmds_legs = np.zeros([len(times), len(joints_commanded_legs)])

    # Video recording
    if clargs.record:
        record_data = np.zeros([len(times)//25, 480, 640, 4], dtype=np.uint8)

    # Run simulation
    while sim_step < len(times):
        if pybullet.readUserDebugParameter(play_id) < 0.5:
            time.sleep(0.5)
        else:
            tic_rt = time.time()
            sim_time = timestep*sim_step
            # Control
            new_freq = pybullet.readUserDebugParameter(freq_id)
            new_body_offset = pybullet.readUserDebugParameter(body_offset_id)
            new_gait = (
                "standing"
                if pybullet.readUserDebugParameter(gait_id) < 0.5
                else "walking"
                if 0.5 < pybullet.readUserDebugParameter(gait_id) < 1.5
                else "swimming"
            )
            if frequency != new_freq:
                gait = new_gait
                frequency = new_freq
                controller.update_frequency(frequency)
            if body_offset != new_body_offset:
                gait = new_gait
                body_offset = new_body_offset
                controller.update_body_offset(body_offset)
            if gait != new_gait:
                gait = new_gait
                frequency = new_freq
                controller = SalamanderController.gait(
                    robot,
                    joints,
                    gait
                )
                pybullet.setGravity(0, 0, -1e-2 if gait == "swimming" else -9.81)
            tic_control = time.time()
            controller.control()
            time_control = time.time() - tic_control
            # Swimming
            if gait == "swimming":
                forces_torques[sim_step] = viscous_swimming(robot, links)
            # Time plugins
            time_plugin = time.time() - tic_rt
            # Physics
            tic_sim = time.time()
            pybullet.stepSimulation()
            sim_step += 1
            toc_sim = time.time()
            tot_sim_time += toc_sim - tic_sim
            # Contacts during walking
            _, contact_forces[sim_step-1, :] = get_links_contacts(
                robot,
                [links[foot] for foot in feet],
                plane
            )
            # Force_torque sensors during walking
            feet_ft[sim_step-1, :, :] = get_joints_force_torque(
                robot,
                [joints[joint] for joint in joints_sensors]
            )
            # Commands
            joints_cmds_body[sim_step-1, :] = get_joints_commands(
                robot,
                [joints[joint] for joint in joints_commanded_body]
            )
            joints_cmds_legs[sim_step-1, :] = get_joints_commands(
                robot,
                [joints[joint] for joint in joints_commanded_legs]
            )
            # Video recording
            if clargs.record and not sim_step % 25:
                record_data[sim_step//25-1, :, :] = record_camera(
                    position=target_pos,
                    yaw=sim_time*360/10,
                    pitch=-30,
                    distance=1
                )
            # User camera
            if not clargs.free_camera:
                target_pos = camera_view(
                    robot,
                    target_pos,
                    yaw_speed=360/10 if clargs.rotating_camera else 0,
                    timestep=timestep
                )
            # Real-time
            toc_rt = time.time()
            rtl = pybullet.readUserDebugParameter(rtl_id)
            if not clargs.fast and rtl < 3:
                real_time_handing(
                    timestep, tic_rt, toc_rt,
                    rtl=rtl,
                    time_plugin=time_plugin,
                    time_sim=toc_sim-tic_sim,
                    time_control=time_control
                )
        keys = pybullet.getKeyboardEvents()
        if ord("q") in keys:
            break

    toc = time.time()

    # Plot contacts
    plt.figure("Contacts")
    for foot_i, foot in enumerate(feet):
        plt.plot(times, contact_forces[:, foot_i], label=foot)
        plt.xlabel("Time [s]")
        plt.ylabel("Reaction force [N]")
        plt.grid(True)
        plt.legend()

    # Plot Feet forces
    plt.figure("Feet forces")
    for dim in range(3):
        plt.plot(times, feet_ft[:, 0, dim], label=["x", "y", "z"][dim])
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.grid(True)
        plt.legend()

    # Plot Feet forces
    plt.figure("Body motor torques")
    for joint_i, joint in enumerate(joints_commanded_body):
        plt.plot(times, joints_cmds_body[:, joint_i], label=joint)
        plt.xlabel("Time [s]")
        plt.ylabel("Torque [Nm]")
        plt.grid(True)
        plt.legend()
    plt.figure("Legs motor torques")
    for joint_i, joint in enumerate(joints_commanded_legs):
        plt.plot(times, joints_cmds_legs[:, joint_i], label=joint)
        plt.xlabel("Time [s]")
        plt.ylabel("Torque [Nm]")
        plt.grid(True)
        plt.legend()

    # Show plots
    plt.show()

    sim_time = timestep*(sim_step)
    print("Time to simulate {} [s]: {} [s] ({} [s] in Bullet)".format(
        sim_time,
        toc-tic,
        tot_sim_time
    ))

    pybullet.disconnect()

    # Record video
    if clargs.record:
        import cv2
        writer = cv2.VideoWriter(
            'test1.avi',
            cv2.VideoWriter_fourcc(*'MJPG'),
            40,
            (640, 480)
        )
        for data in record_data:
            writer.write(data)


def main_parallel():
    """Simulation with multiprocessing"""
    from multiprocessing import Pool

    # Create Pool
    p = Pool(1)

    # Parse command line arguments
    clargs = parse_args()

    # Run simulation
    p.map(main, [clargs])
    print("Done")


if __name__ == '__main__':
    # main2(parse_args())
    main(parse_args())
