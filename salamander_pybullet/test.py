"""Salamander simulation with pybullet"""

import time
import argparse

import numpy as np

import pybullet_data
import pybullet


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
        help='Enable rotating_camera'
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
        self.angular_frequency = 2*np.pi*frequency
        self.phase = phase
        self.offset = offset

    def position(self, sim_time):
        """"Position"""
        return self.amplitude*np.sin(
            self.angular_frequency*sim_time + self.phase
        ) + self.offset

    def velocity(self, sim_time):
        """Velocity"""
        return self.angular_frequency*self.amplitude*np.cos(
            self.angular_frequency*sim_time + self.phase
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

    def __init__(self, joint, sine, pdf):
        super(JointController, self).__init__()
        self._joint = joint
        self._sine = sine
        self._pdf = pdf

    def cmds(self, sim_time):
        """Commands"""
        return {
            "pos": self._sine.position(sim_time),
            "vel": self._sine.velocity(sim_time)
        }

    def pdf_terms(self):
        """pdf"""
        return self._pdf

    def update(self, sim_time):
        """Update"""
        return {
            "joint": self._joint,
            "cmd": self.cmds(sim_time),
            "pdf": self.pdf_terms()
        }


class RobotController:
    """RobotController"""

    def __init__(self, robot, joints_controllers):
        super(RobotController, self).__init__()
        self.robot = robot
        self.controllers = joints_controllers

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
                )
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
                    amplitude= (
                        float(0.6 if joint_i == 0 else 0.1)
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
                            - (0 if joint_i == 0 else 0.5*np.pi)
                        )
                        if gait == "walking"
                        else 0
                    ),
                    offset=(
                        float(0 if joint_i == 0 else 0.1)
                        if gait == "walking"
                        else -2*np.pi/5 if joint_i == 0
                        else 0
                    )
                ),
                pdf=ControlPDF(p=1e-1, d=1e0, f=1e1)
            )
            for leg_i in range(2)
            for side_i, side in enumerate(["L", "R"])
            for joint_i in range(3)
        ]
        return cls(robot, joint_controllers_body+joint_controllers_legs)

    def control(self, sim_time):
        """Control"""
        controls = [
            controller.update(sim_time)
            for controller in self.controllers
        ]
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


def init_engine():
    """Initialise engine"""
    print(pybullet.getAPIVersion())
    pybullet.connect(pybullet.GUI)
    pybullet_path = pybullet_data.getDataPath()
    print("Adding pybullet data path {}".format(pybullet_path))
    pybullet.setAdditionalSearchPath(pybullet_path)


def init_physics(time_step, gait="walking"):
    """Initialise physics"""
    pybullet.resetSimulation()
    pybullet.setGravity(0, 0, -9.81 if gait == "walking" else 0)
    pybullet.setTimeStep(time_step)
    pybullet.setRealTimeSimulation(0)
    pybullet.setPhysicsEngineParameter(
        fixedTimeStep=1e-3,
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
    distance = kwargs.pop("distance", 1)
    yaw = kwargs.pop("yaw", 0)
    pitch = kwargs.pop("pitch", -45)
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
    """Viscous swimming

    TODO: CORRECT LOCAL/WORLD FRAME

    """
    # Swimming
    for link_i in range(1, 11):
        link_state = pybullet.getLinkState(
            robot,
            links["link_body_{}".format(link_i+1)],
            computeLinkVelocity=1,
            computeForwardKinematics=1
        )
        link_velocity = link_state[6]
        link_angular_velocity = link_state[7]
        pybullet.applyExternalForce(
            robot,
            links["link_body_{}".format(link_i+1)],
            forceObj=[
                -1e-2*link_velocity[0],
                -1e-1*link_velocity[1],
                -1e-1*link_velocity[2],
            ],
            posObj=[0, 0, 0],
            flags=pybullet.LINK_FRAME
        )
        pybullet.applyExternalTorque(
            robot,
            links["link_body_{}".format(link_i+1)],
            forceObj=[
                link_angular_velocity[0],
                link_angular_velocity[1],
                link_angular_velocity[2]
            ],
            posObj=[0, 0, 0],
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


def init_simulation(time_step, gait="walking"):
    """Initialise simulation"""
    # Physics
    init_physics(time_step, gait)

    # Spawn models
    robot, _ = spawn_models()

    # Links and joints
    links, joints, _ = get_joints(robot)
    print("Links ids:\n{}".format(links))
    print("Joints ids:\n{}".format(joints))
    return robot, links, joints


def user_parameters():
    """User parameters"""
    rtl_id = pybullet.addUserDebugParameter(
        paramName="Real-time limiter",
        rangeMin=1e-3,
        rangeMax=3,
        startValue=1
    )
    pybullet.addUserDebugParameter(
        paramName="Gait",
        rangeMin=0,
        rangeMax=1,
        startValue=0
    )
    freq_id = pybullet.addUserDebugParameter(
        paramName="Frequency",
        rangeMin=0,
        rangeMax=3,
        startValue=1
    )
    for part in ["body", "legs"]:
        for pdf in ["p", "d", "f"]:
            pybullet.addUserDebugParameter(
                paramName="{}_{}".format(part, pdf),
                rangeMin=0,
                rangeMax=10,
                startValue=0.1
            )
    return freq_id, rtl_id


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


def main():
    """Main"""
    # Parse command line arguments
    clargs = parse_args()

    # Initialise engine
    init_engine()

    # Parameters
    gait = "walking"
    # gait = "swimming"
    time_step = 1e-3

    # Initialise
    robot, _, joints = init_simulation(time_step, gait)

    # Controller
    frequency = 1 if gait == "walking" else 2
    controller = RobotController.salamander(
        robot,
        joints,
        gait=gait,
        frequency=frequency
    )

    # Camera
    target_pos = camera_view(robot)

    # User parameters
    freq_id, rtl_id = user_parameters()

    # Video recording
    record = False

    # Debug info
    test_debug_info()

    # Run simulation
    tic = time.time()
    tot_sim_time = 0
    for sim_step in range(int(10/time_step)):
        tic_rt = time.time()
        sim_time = time_step*sim_step
        # Control
        new_freq = pybullet.readUserDebugParameter(freq_id)
        if frequency != new_freq:
            frequency = new_freq
            controller = RobotController.salamander(
                robot,
                joints,
                gait=gait,
                frequency=frequency
            )
        controller.control(sim_time)
        # Swimming
        if gait == "swimming":
            viscous_swimming(robot, links)
        # Physics
        tic_sim = time.time()
        pybullet.stepSimulation()
        toc_sim = time.time()
        tot_sim_time += toc_sim - tic_sim
        # Video recording
        if record and not sim_step % 30:
            yaw = sim_time*360/10 if clargs.rotating_camera else 0
            record_camera(
                position=pybullet.getBasePositionAndOrientation(robot)[0],
                yaw=yaw
            )
        # User camera
        if not clargs.free_camera:
            yaw = sim_time*360/10 if clargs.rotating_camera else 0
            target_pos = camera_view(robot, target_pos, yaw=yaw)
        # Real-time
        toc_rt = time.time()
        if not clargs.fast:
            rtl = pybullet.readUserDebugParameter(rtl_id)
            sleep_rtl = time_step/rtl - (toc_rt - tic_rt)
            rtf = time_step / (toc_rt - tic_rt)
            if sleep_rtl > 0:
                time.sleep(sleep_rtl)
            if rtf < 1:
                print("Slower than real-time: {} %".format(100*rtf))
    toc = time.time()

    sim_time = time_step*(sim_step+1)
    print("Time to simulate {} [s]: {} [s] ({} [s] in Bullet)".format(
        sim_time,
        toc-tic,
        tot_sim_time
    ))

    pybullet.disconnect()


if __name__ == '__main__':
    main()
