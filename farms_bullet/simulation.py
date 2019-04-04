"""Salamander simulation with pybullet"""

import time

import numpy as np
import matplotlib.pyplot as plt

import pybullet

from .plugins.swimming import viscous_swimming
from .simulator import init_engine, real_time_handing
from .debug import test_debug_info
from .arenas import create_scene
from .render import rendering
from .model import Model, SalamanderModel
from .interface import UserParameters
from .camera import UserCamera, CameraRecord
from .logging import ExperimentLogger
from .simulation_options import SimulationOptions
from .model_options import ModelOptions


class SimulationElement:
    """Documentation for SimulationElement"""

    def __init__(self):
        super(SimulationElement, self).__init__()
        self._identity = None

    @property
    def identity(self):
        """Element identity"""
        return self._identity

    @staticmethod
    def spawn():
        """Spawn"""

    @staticmethod
    def step():
        """Step"""

    @staticmethod
    def log():
        """Log"""

    @staticmethod
    def save_logs():
        """Save logs"""

    @staticmethod
    def plot():
        """Plot"""

    @staticmethod
    def reset():
        """Reset"""

    @staticmethod
    def delete():
        """Delete"""


class Animat(SimulationElement):
    """Animat"""

    def __init__(self, options):
        super(Animat, self).__init__()
        self.options = options


class Salamander(Animat):
    """Salamander animat"""

    def __init__(self, options, timestep, n_iterations):
        super(Salamander, self).__init__(options)
        self.model = None
        self.timestep = timestep
        self.logger = None
        self.n_iterations = n_iterations

    def spawn(self):
        """Spawn"""
        self.model = SalamanderModel.spawn(
            self.timestep,
            **self.options
        )
        self._identity = self.model.identity
        self.logger = ExperimentLogger(
            self.model,
            self.n_iterations
        )

    @property
    def links(self):
        """Links"""
        return self.model.links

    @property
    def joints(self):
        """Joints"""
        return self.model.joints

    def step(self):
        """Step"""
        self.animat_physics()
        self.animat_control()

    def log(self):
        """Log"""
        self.animat_logging()

    def animat_physics(self):
        """Animat physics"""
        # Swimming
        forces = None
        if self.options.gait == "swimming":
            forces = viscous_swimming(
                self.identity,
                self.links
            )
        return forces

    def animat_control(self):
        """Control animat"""
        # Control
        tic_control = time.time()
        self.model.controller.control()
        time_control = time.time() - tic_control
        return time_control

    def animat_logging(self, sim_step):
        """Animat logging"""
        # Contacts during walking
        tic_sensors = time.time()
        self.model.sensors.update(
            identity=self.identity,
            links=[self.links[foot] for foot in self.model.feet],
            joints=[
                self.joints[joint]
                for joint in self.model.sensors.joints_sensors
            ]
        )
        # Commands
        self.model.motors.update(
            identity=self.identity,
            joints_body=[
                self.joints[joint]
                for joint in self.model.motors.joints_commanded_body
            ],
            joints_legs=[
                self.joints[joint]
                for joint in self.model.motors.joints_commanded_legs
            ]
        )
        time_sensors = time.time() - tic_sensors
        tic_log = time.time()
        self.logger.update(sim_step-1)
        time_log = time.time() - tic_log
        return time_sensors, time_log


class AnimatLink:
    """Animat link"""

    def __init__(self, **kwargs):
        super(AnimatLink, self).__init__()
        self.size = kwargs.pop("size", [0.1, 0.1, 0.1])
        self.geometry = kwargs.pop("geometry", pybullet.GEOM_BOX)
        self.position = kwargs.pop("position", [0, 0, 0])
        self.orientation = pybullet.getQuaternionFromEuler(
            kwargs.pop("orientation", [0, 0, 0])
        )
        self.f_position = kwargs.pop("f_position", [0, 0, 0])
        self.f_orientation = pybullet.getQuaternionFromEuler(
            kwargs.pop("f_orientation", [0, 0, 0])
        )
        self.mass = kwargs.pop("mass", 0)
        self.parent = kwargs.pop("parent", None)
        self.collision = pybullet.createCollisionShape(
            self.geometry,
            halfExtents=self.size,
            collisionFramePosition=self.position,
            collisionFrameOrientation=self.orientation
        )
        color = kwargs.pop("color", None)
        self.visual = -1 if color is None else pybullet.createVisualShape(
            self.geometry,
            halfExtents=self.size,
            visualFramePosition=self.position,
            visualFrameOrientation=self.orientation,
            rgbaColor=color
        )

        # Joint
        self.joint_type = kwargs.pop("joint_type", pybullet.JOINT_REVOLUTE)
        self.joint_axis = kwargs.pop("joint_axis", [0, 0, 1])


class SimonAnimat(Animat):
    """Documentation for SimonAnimat"""

    def __init__(self, options, timestep, n_iterations):
        super(SimonAnimat, self).__init__(options)
        self.timestep = timestep
        self.n_iterations = n_iterations
        self.logger = None

    def spawn(self):
        """Spawn"""
        print("Spawning animat")
        base_link = AnimatLink(
            size=[0.1, 0.05, 0.03],
            geometry=pybullet.GEOM_BOX,
            position=[0, 0, 0],
            orientation=[0, 0, 0],
            f_position=[0, 0, 0],
            mass=1
        )
        # Upper legs
        upper_legs_positions = np.array([
            [0.05, 0.04, -0.02],
            [0.05, -0.04, -0.02],
            [-0.05, 0.04, -0.02],
            [-0.05, -0.04, -0.02]
        ])
        upper_legs = [
            AnimatLink(
                size=[0.02, 0.02, 0.04],
                geometry=pybullet.GEOM_BOX,
                position=position,
                orientation=[0, 0, 0],
                f_position=[0, 0, 0],
                f_orientation=[0, 0, 0],
                joint_axis=[1, 0, 0],
                mass=0.1
            ) for position in upper_legs_positions
        ]
        upper_legs[0].parent = 0
        upper_legs[1].parent = 0
        upper_legs[2].parent = 0
        upper_legs[3].parent = 0
        # Lower legs
        lower_legs = [
            AnimatLink(
                size=[0.02, 0.02, 0.04],
                geometry=pybullet.GEOM_BOX,
                position=upper_pos, # + np.array([0, 0, -0.04]),
                orientation=[0, 0, 0],
                f_position=[0, 0, 0],
                f_orientation=[0, 0, 0],
                joint_axis=[1, 0, 0],
                mass=0.1
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
            basePosition=[0, 0, 0.5],
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
        n_joints = pybullet.getNumJoints(self.identity)
        for joint in range(n_joints):
            pybullet.changeDynamics(
                self.identity,
                joint,
                spinningFriction=0.1,
                rollingFriction=0.1,
                linearDamping=0.1,
                jointDamping=0.05
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
            print(pybullet.getJointInfo(self.identity, i))


class Floor(SimulationElement):
    """Floor"""

    def __init__(self, position):
        super(Floor, self).__init__()
        self._position = position
        self.model = None

    def spawn(self):
        """Spawn floor"""
        self.model = Model.from_urdf(
            "plane.urdf",
            basePosition=self._position
        )
        self._identity = self.model.identity


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


class Interfaces:
    """Interfaces (GUI, camera, video)"""

    def __init__(self, camera=None, user_params=None, video=None):
        super(Interfaces, self).__init__()
        self.camera = camera
        self.user_params = user_params
        self.video = video
        self.camera_skips = 10

    def init_camera(self, target_identity, timestep, **kwargs):
        """Initialise camera"""
        # Camera
        self.camera = UserCamera(
            target_identity=target_identity,
            yaw=0,
            yaw_speed=(
                360/10*self.camera_skips
                if kwargs.pop("rotating_camera", False)
                else 0
            ),
            pitch=-89 if kwargs.pop("top_camera", False) else -45,
            distance=1,
            timestep=timestep
        )

    def init_video(self, target_identity, timestep, size, **kwargs):
        """Init video"""
        # Video recording
        self.video = CameraRecord(
            target_identity=target_identity,
            size=size,
            fps=kwargs.pop("fps", 40),
            yaw=kwargs.pop("yaw", 0),
            yaw_speed=360/10 if kwargs.pop("rotating_camera", False) else 0,
            pitch=-89 if kwargs.pop("top_camera", False) else -45,
            distance=1,
            timestep=timestep,
            motion_filter=1e-1
        )

    def init_debug(self, animat_options):
        """Initialise debug"""
        # User parameters
        self.user_params = UserParameters(
            animat_options.gait,
            animat_options.frequency
        )

        # Debug info
        test_debug_info()


class SimulationProfiler:
    """Simulation profiler"""

    def __init__(self, sim_duration):
        super(SimulationProfiler, self).__init__()
        self.sim_duration = sim_duration
        self.plugin_time = 0
        self.sim_time = 0
        self.physics_time = 0
        self.ctrl_time = 0
        self.sensors_time = 0
        self.log_time = 0
        self.camera_time = 0
        self.waitrt_time = 0

    def reset(self):
        """Reset"""
        self.plugin_time = 0
        self.sim_time = 0
        self.physics_time = 0
        self.ctrl_time = 0
        self.sensors_time = 0
        self.log_time = 0
        self.camera_time = 0
        self.waitrt_time = 0

    def total_time(self):
        """Total time"""
        return (
            self.plugin_time
            + self.physics_time
            + self.ctrl_time
            + self.sensors_time
            + self.log_time
            + self.camera_time
            + self.waitrt_time
        )

    def print_times(self):
        """Print times"""
        print("Time to simulate {} [s]: {} [s]".format(
            self.sim_duration,
            self.sim_time,
        ))
        print("  Plugin: {} [s]".format(self.plugin_time))
        print("  Bullet physics: {} [s]".format(self.physics_time))
        print("  Controller: {} [s]".format(self.ctrl_time))
        print("  Sensors: {} [s]".format(self.sensors_time))
        print("  Logging: {} [s]".format(self.log_time))
        print("  Camera: {} [s]".format(self.camera_time))
        print("  Wait real-time: {} [s]".format(self.waitrt_time))
        print("  Sum: {} [s]".format(
            self.plugin_time
            + self.physics_time
            + self.ctrl_time
            + self.sensors_time
            + self.log_time
            + self.camera_time
            + self.waitrt_time
        ))


class Experiment:
    """Experiment"""

    def __init__(self, animat, arena, timestep, n_iterations):
        super(Experiment, self).__init__()
        self.animat = animat
        self.arena = arena
        self.timestep = timestep
        self.n_iterations = n_iterations
        self.logger = None

    def elements(self):
        """Elements in experiment"""
        return [self.animat, self.arena]

    def _spawn(self):
        """Spawn"""
        for element in self.elements():
            element.spawn()

    def spawn(self):
        """Spawn"""
        self._spawn()

    def step(self):
        """Step"""
        for element in self.elements():
            element.step()

    def log(self):
        """Step"""
        for element in self.elements():
            element.log()


class SalamanderExperiment(Experiment):
    """Salamander experiment"""

    def __init__(self, sim_options, n_iterations, **kwargs):
        self.animat_options = kwargs.pop("animat_options", ModelOptions())
        super(SalamanderExperiment, self).__init__(
            animat=Salamander(
                self.animat_options,
                sim_options.timestep,
                n_iterations
            ),
            arena=FlooredArena(),
            timestep=sim_options.timestep,
            n_iterations=n_iterations
        )
        self.sim_options = sim_options
        self.interface = Interfaces()
        self.simulation_state = None
        self.profile = SimulationProfiler(self.sim_options.duration)
        self.forces_torques = np.zeros([n_iterations, 2, 10, 3])

    def spawn(self):
        """Spawn"""
        # Elements
        self._spawn()
        self.animat.model.sensors.plane = self.arena.floor.identity
        # Interface
        if not self.sim_options.headless:
            self.interface.init_camera(
                target_identity=self.animat.identity,
                timestep=self.timestep,
                rotating_camera=self.sim_options.rotating_camera,
                top_camera=self.sim_options.top_camera
            )
            self.interface.init_debug(animat_options=self.animat_options)
        if self.sim_options.record and not self.sim_options.headless:
            self.interface.init_video(
                target_identity=self.animat.identity,
                timestep=self.timestep*25,
                size=self.n_iterations//25,
                rotating_camera=self.sim_options.rotating_camera,
                top_camera=self.sim_options.top_camera
            )

    def save(self):
        """Save experiment state"""
        self.simulation_state = pybullet.saveState()

    def pre_step(self, sim_step):
        """New step"""
        play = self.interface.user_params.play.value
        if not sim_step % 100:
            self.interface.user_params.update()
        if not(sim_step % 10000) and sim_step > 0:
            pybullet.restoreState(self.simulation_state)
        if not play:
            time.sleep(0.5)
            self.interface.user_params.update()
        return play

    def step(self, sim_step):
        """Simulation step"""
        self.tic_rt = time.time()
        self.sim_time = self.timestep*sim_step

        # Time plugins
        self.animat_interface()
        external_forces = self.animat.animat_physics()
        if external_forces is not None:
            self.forces_torques[sim_step] = external_forces
        self.time_plugin = time.time() - self.tic_rt
        self.profile.plugin_time += self.time_plugin
        # Control animat
        time_control = self.animat.animat_control()
        self.profile.ctrl_time += time_control
        # Physics
        self.tic_sim = time.time()
        pybullet.stepSimulation()
        self.toc_sim = time.time()
        sim_step += 1
        self.profile.physics_time += self.toc_sim - self.tic_sim
        # Animat logging
        time_sensors, time_log = self.animat.animat_logging(sim_step)
        self.profile.sensors_time += time_sensors
        self.profile.log_time += time_log
        # Camera
        tic_camera = time.time()
        if not self.sim_options.headless:
            if self.sim_options.record and not sim_step % 25:
                self.camera_record.record(sim_step//25-1)
            # User camera
            if (
                    not sim_step % self.interface.camera_skips
                    and not self.sim_options.free_camera
            ):
                self.interface.camera.update()
        self.profile.camera_time += time.time() - tic_camera
        # Real-time
        self.toc_rt = time.time()
        tic_rt = time.time()
        if not self.sim_options.fast and self.interface.user_params.rtl.value < 3:
            real_time_handing(
                self.timestep, self.tic_rt, self.toc_rt,
                rtl=self.interface.user_params.rtl.value,
                time_plugin=self.time_plugin,
                time_sim=self.toc_sim-self.tic_sim,
                time_control=time_control
            )
        self.profile.waitrt_time = time.time() - tic_rt

    def animat_interface(self):
        """Animat interface"""
        # Control
        if self.interface.user_params.gait.changed:
            self.animat_options.gait = self.interface.user_params.gait.value
            self.animat.model.controller.update_gait(
                self.animat_options.gait,
                self.animat.joints,
                self.timestep
            )
            pybullet.setGravity(
                0, 0, -1e-2 if self.animat_options.gait == "swimming" else -9.81
            )
            self.interface.user_params.gait.changed = False
        if self.interface.user_params.frequency.changed:
            self.animat.model.controller.update_frequency(
                self.interface.user_params.frequency.value
            )
            self.interface.user_params.frequency.changed = False
        if self.interface.user_params.body_offset.changed:
            self.animat.model.controller.update_body_offset(
                self.interface.user_params.body_offset.value
            )
            self.interface.user_params.body_offset.changed = False

    def postprocess(self):
        """Plot after simulation"""
        # Plot
        self.animat.logger.plot_all(self.times_simulated)
        plt.show()

        # Record video
        if self.sim_options.record and not self.sim_options.headless:
            self.camera_record.save("video.avi")

    def end(self, sim_step, sim_time):
        """Terminate experiment"""
        self.profile.sim_duration = self.timestep*sim_step
        self.profile.sim_time = sim_time
        self.profile.print_times()


class SimonExperiment(Experiment):
    """Simon experiment"""

    def __init__(self, sim_options, n_iterations, **kwargs):
        self.animat_options = kwargs.pop("animat_options", ModelOptions())
        super(SimonExperiment, self).__init__(
            animat=SimonAnimat(
                self.animat_options,
                sim_options.timestep,
                n_iterations
            ),
            arena=FlooredArena(),
            timestep=sim_options.timestep,
            n_iterations=n_iterations
        )

    def pre_step(self, sim_step):
        """New step"""
        return True

    def step(self, sim_step):
        """Step"""
        n_joints = pybullet.getNumJoints(self.animat.identity)
        # pybullet.setJointMotorControlArray(
        #     self.animat.identity,
        #     np.arange(n_joints),
        #     pybullet.TORQUE_CONTROL,
        #     forces=0.2*np.ones(n_joints)
        # )
        target_positions = np.zeros(n_joints)
        target_velocities = np.zeros(n_joints)
        joint_control = int((1e-3 * sim_step) % n_joints)
        target_positions[joint_control] = (
            0.1 if 200 < sim_step % 1000 < 500
            else -0.1 if 500 < sim_step % 1000 < 800
            else 0
        )
        pybullet.setJointMotorControlArray(
            self.animat.identity,
            np.arange(n_joints),
            pybullet.POSITION_CONTROL,
            targetPositions=target_positions,
            targetVelocities=target_velocities,
            forces=np.ones(n_joints)
        )
        pybullet.stepSimulation()
        sim_step += 1
        return time.sleep(1e-3)


class Simulation:
    """Simulation"""

    def __init__(self, experiment, simulation_options, animat_options):
        super(Simulation, self).__init__()

        # Options
        self.sim_options = simulation_options
        self.animat_options = animat_options

        # Initialise engine
        init_engine(self.sim_options.headless)
        rendering(0)

        # Parameters
        self.timestep = self.sim_options.timestep
        self.times = np.arange(0, self.sim_options.duration, self.timestep)

        # Initialise physics
        self.init_physics()

        # Initialise models
        self.experiment = experiment
        self.experiment.spawn()
        self.animat = self.experiment.animat

        # Simulation
        self.sim_step = 0

        rendering(1)

    def init_physics(self):
        """Initialise physics"""
        gait = self.animat_options.gait
        pybullet.resetSimulation()
        pybullet.setGravity(0, 0, -1e-2 if gait == "swimming" else -9.81)
        pybullet.setTimeStep(self.timestep)
        pybullet.setRealTimeSimulation(0)
        pybullet.setPhysicsEngineParameter(
            fixedTimeStep=self.timestep,
            numSolverIterations=50,
            erp=0,
            contactERP=0,
            frictionERP=0
        )
        print("Physics parameters:\n{}".format(
            pybullet.getPhysicsEngineParameters()
        ))

    def run(self):
        """Run simulation"""
        # Run simulation
        self.tic = time.time()
        loop_time = 0
        play = True
        while self.sim_step < len(self.times):
            if not self.sim_options.headless:
                keys = pybullet.getKeyboardEvents()
                if ord("q") in keys:
                    break
                play = self.experiment.pre_step(self.sim_step)
            if play:
                tic_loop = time.time()
                self.experiment.step(self.sim_step)
                self.sim_step += 1
                # self.experiment.log()
                loop_time += time.time() - tic_loop
        print("Loop time: {} [s]".format(loop_time))
        self.toc = time.time()
        self.experiment.times_simulated = self.times[:self.sim_step]

    def end(self):
        """Terminate simulation"""
        # End experiment
        self.experiment.end(self.sim_step, self.toc - self.tic)
        # Disconnect from simulation
        pybullet.disconnect()


class SalamanderSimulation(Simulation):
    """Salamander simulation"""

    def __init__(self, simulation_options, animat_options):
        experiment = SalamanderExperiment(
            simulation_options,
            len(np.arange(
                0, simulation_options.duration, simulation_options.timestep
            )),
            animat_options=animat_options
        )
        super(SalamanderSimulation, self).__init__(
            experiment=experiment,
            simulation_options=simulation_options,
            animat_options=animat_options
        )

        self.animat.model.leg_collisions(
            self.experiment.arena.floor.identity,
            activate=False
        )
        self.animat.model.print_dynamics_info()


class SimonSimulation(Simulation):
    """Simon experiment simulation"""

    def __init__(self, simulation_options, animat_options):
        experiment = SimonExperiment(
            simulation_options,
            len(np.arange(
                0, simulation_options.duration, simulation_options.timestep
            )),
            animat_options=animat_options
        )
        super(SimonSimulation, self).__init__(
            experiment=experiment,
            simulation_options=simulation_options,
            animat_options=animat_options
        )


def run_simon(sim_options=None, animat_options=None):
    """Run Simon's experiment"""

    # Parse command line arguments
    if not sim_options:
        simulation_options = SimulationOptions.with_clargs(duration=100)
    if not animat_options:
        animat_options = ModelOptions()

    # Setup simulation
    print("Creating simulation")
    sim = SimonSimulation(
        simulation_options=simulation_options,
        animat_options=animat_options
    )

    # Run simulation
    print("Running simulation")
    sim.run()

    # # Show results
    # print("Analysing simulation")
    # sim.experiment.postprocess()
    # sim.end()


def main(sim_options=None, animat_options=None):
    """Main"""

    # Parse command line arguments
    if not sim_options:
        simulation_options = SimulationOptions.with_clargs()
    if not animat_options:
        animat_options = ModelOptions()

    # Setup simulation
    print("Creating simulation")
    sim = SalamanderSimulation(
        simulation_options=simulation_options,
        animat_options=animat_options
    )

    # Run simulation
    print("Running simulation")
    sim.run()

    # Show results
    print("Analysing simulation")
    sim.experiment.postprocess()
    sim.end()


def main_parallel():
    """Simulation with multiprocessing"""
    from multiprocessing import Pool

    # Parse command line arguments
    sim_options = SimulationOptions.with_clargs()

    # Create Pool
    pool = Pool(2)

    # Run simulation
    pool.map(main, [sim_options, sim_options])
    print("Done")


if __name__ == '__main__':
    # main_parallel()
    main()
