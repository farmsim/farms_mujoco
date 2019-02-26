"""Salamander simulation with pybullet"""

import time

import numpy as np

import pybullet_data
import pybullet


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
                    if gait == "walking"
                    else ControlPDF(p=1e-1, d=1e0, f=1e1)
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


def init_physics(time_step):
    """Initialise physics"""
    pybullet.resetSimulation()
    pybullet.setGravity(0, 0, -9.81)
    pybullet.setTimeStep(time_step)
    pybullet.setRealTimeSimulation(0)


def spawn_models():
    """Spawn models"""
    robot = pybullet.loadSDF(
        "/home/jonathan/.gazebo/models/biorob_salamander/model.sdf"
    )[0]
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
    joints = {
        info[1].decode("UTF-8"): info[0]
        for info in [
            pybullet.getJointInfo(robot, j)
            for j in range(n_joints)
        ]
    }
    return joints, n_joints


def main():
    """Main"""
    init_engine()

    time_step = 1e-3
    init_physics(time_step)

    robot, _ = spawn_models()

    joints, _ = get_joints(robot)
    print("Joints ids:\n{}".format(joints))

    # Controller
    gait = "walking"
    # gait = "swimming"
    controller = RobotController.salamander(robot, joints, gait=gait)

    # Camera
    camera_filter = 1e-3
    targetPos = np.array(pybullet.getBasePositionAndOrientation(robot)[0])

    # PDF parameters
    pybullet.addUserDebugParameter(
        paramName="Gait",
        rangeMin=0,
        rangeMax=1,
        startValue=0
    )
    pybullet.addUserDebugParameter(
        paramName="pfd_p",
        rangeMin=0,
        rangeMax=10,
        startValue=0.1
    )

    tic = time.time()
    for sim_step in range(int(10/time_step)):
        tic_rt = time.time()
        sim_time = time_step*sim_step
        controller.control(sim_time)
        # control_robot(robot, joints, sim_time)
        pybullet.stepSimulation()
        distance, yaw, pitch = 1, 0, -45
        targetPos = (
            (1-camera_filter)*targetPos
            + camera_filter*np.array(
                pybullet.getBasePositionAndOrientation(robot)[0]
            )
        )
        pybullet.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=targetPos
        )
        toc_rt = time.time()
        sleep_rt = time_step - (toc_rt - tic_rt)
        if sleep_rt > 0:
            time.sleep(sleep_rt)
    toc = time.time()

    sim_time = time_step*(sim_step+1)
    print("Time to simulate {} [s]: {} [s]".format(sim_time, toc-tic))

    pybullet.disconnect()


if __name__ == '__main__':
    main()
