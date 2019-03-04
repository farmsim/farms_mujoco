"""Salamander simulation with pybullet"""

import time
import argparse

import numpy as np

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

    @classmethod
    def salamander(cls, robot, joints, **kwargs):
        """Salamander controller"""
        n_body_joints = kwargs.pop("n_body_joints", 11)
        gait = kwargs.pop("gait", "walking")
        frequency = kwargs.pop("frequency", 1)
        joint_controllers_body = [
            JointController(
                joint=joints["joint_link_body_{}".format(joint_i+1)],
                sine=SineControl(
                    amplitude=(
                        0.2*np.sin(2*np.pi*joint_i/n_body_joints - np.pi/4)
                        if gait == "walking"
                        else 0.1+joint_i*0.4/n_body_joints
                    ),
                    frequency=frequency,
                    phase=(
                        0
                        if gait == "walking"
                        else -2*np.pi*joint_i/11
                    ),
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
                        float(
                            0.8
                            if joint_i == 0
                            else np.pi/16*leg_i if joint_i == 1
                            else np.pi/8
                        )
                        if gait == "walking"
                        else 0.0
                    ),
                    frequency=(
                        float(frequency)
                        if gait == "walking"
                        else 0
                    ),
                    phase=(
                        float(
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
                        )
                        if gait == "walking"
                        else 0
                    ),
                    offset=(
                        float(
                            0
                            if joint_i == 0
                            else np.pi/16*leg_i if joint_i == 1
                            else np.pi/8
                        )
                        if gait == "walking"
                        else (-2*np.pi/5 if joint_i == 0 else 0)
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
    pybullet.setGravity(0, 0, -9.81 if gait == "walking" else -1e-2)
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
        info[12].decode("UTF-8"): info[16]
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
    for link_i in range(1, 11):
        link_state = pybullet.getLinkState(
            robot,
            links["link_body_{}".format(link_i+1)],
            computeLinkVelocity=1,
            computeForwardKinematics=1
        )
        link_orientation = np.linalg.inv(np.array(
            pybullet.getMatrixFromQuaternion(link_state[5])
        ).reshape([3, 3]))
        link_velocity = np.dot(link_orientation, link_state[6])
        link_angular_velocity = np.dot(link_orientation, link_state[7])
        pybullet.applyExternalForce(
            robot,
            links["link_body_{}".format(link_i+1)],
            forceObj=[
                -1e-1*link_velocity[0],
                -1e0*link_velocity[1],
                -1e0*link_velocity[2],
            ],
            posObj=[0, 0, 0],
            flags=pybullet.LINK_FRAME
        )
        pybullet.applyExternalTorque(
            robot,
            links["link_body_{}".format(link_i+1)],
            torqueObj=[
                -1e-2*link_angular_velocity[0],
                -1e-2*link_angular_velocity[1],
                -1e-2*link_angular_velocity[2]
            ],
            flags=pybullet.LINK_FRAME
        )


def record_camera(position, yaw=0):
    """Record camera"""
    image = pybullet.getCameraImage(
        width=800,
        height=480,
        viewMatrix=pybullet.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=position,
            distance=1,
            yaw=yaw,
            pitch=-45,
            roll=0,
            upAxisIndex=2
        ),
        renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
        flags=pybullet.ER_NO_SEGMENTATION_MASK
    )


def init_simulation(timestep, gait="walking"):
    """Initialise simulation"""
    # Physics
    init_physics(timestep, gait)

    # Spawn models
    robot, plane = spawn_models()

    # Links and joints
    links, joints, _ = get_joints(robot)
    print("Links ids:\n{}".format(links))
    print("Joints ids:\n{}".format(joints))
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
        rangeMax=1,
        startValue=0 if gait == "walking" else 1
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


def main():
    """Main"""
    # Parse command line arguments
    clargs = parse_args()

    # Initialise engine
    init_engine()

    # Parameters
    gait = "walking"
    # gait = "swimming"
    timestep = 1e-3

    # Initialise
    robot, links, joints, plane = init_simulation(timestep, gait)

    # Create scene
    add_obstacles = False
    if add_obstacles:
        create_scene(plane)

    # Controller
    frequency = 1 if gait == "walking" else 2
    body_offset = 0
    controller = RobotController.salamander(
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

    # Video recording
    record = False

    # Debug info
    test_debug_info()

    # Run simulation
    tic = time.time()
    tot_sim_time = 0
    times = np.arange(0, 10, timestep)
    forces_torques = np.zeros([len(times), 2, 10, 3])
    sim_step = 0
    while sim_step < len(times) + 1:
        if pybullet.readUserDebugParameter(play_id) < 0.5:
            time.sleep(0.5)
        else:
            tic_rt = time.time()
            sim_time = timestep*sim_step
            # Control
            new_freq = pybullet.readUserDebugParameter(freq_id)
            new_body_offset = pybullet.readUserDebugParameter(body_offset_id)
            new_gait = (
                "walking"
                if pybullet.readUserDebugParameter(gait_id) < 0.5
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
                controller = RobotController.salamander(
                    robot,
                    joints,
                    gait=gait,
                    frequency=frequency
                )
                pybullet.setGravity(0, 0, -9.81 if gait == "walking" else -1e-2)
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
            # Video recording
            if record and not sim_step % 30:
                camera_yaw = sim_time*360/10 if clargs.rotating_camera else 0
                record_camera(
                    position=pybullet.getBasePositionAndOrientation(robot)[0],
                    yaw=camera_yaw
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
    toc = time.time()

    keys = pybullet.getKeyboardEvents()
    print(keys)

    sim_time = timestep*(sim_step+1)
    print("Time to simulate {} [s]: {} [s] ({} [s] in Bullet)".format(
        sim_time,
        toc-tic,
        tot_sim_time
    ))

    pybullet.disconnect()


if __name__ == '__main__':
    main()