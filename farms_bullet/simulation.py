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

    def __init__(self, options, timestep):
        super(Salamander, self).__init__(options)
        self.model = None
        self.timestep = timestep

    def spawn(self):
        """Spawn"""
        self.model = SalamanderModel.spawn(
            self.timestep,
            **self.options
        )
        self._identity = self.model.identity

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

    def log(self):
        """Log"""


class Floor(SimulationElement):
    """Floor"""

    def __init__(self, position):
        super(Floor, self).__init__()
        self._position = position

    def spawn(self):
        """Spawn floor"""
        self._identity = Model.from_urdf(
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
        self.ctrl_time = 0
        self.sensors_time = 0
        self.log_time = 0
        self.camera_time = 0
        self.waitrt_time = 0

    def reset(self):
        """Reset"""
        self.plugin_time = 0
        self.sim_time = 0
        self.ctrl_time = 0
        self.sensors_time = 0
        self.log_time = 0
        self.camera_time = 0
        self.waitrt_time = 0

    def total_time(self):
        """Total time"""
        return (
            self.plugin_time
            + self.sim_time
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
        print("  Bullet physics: {} [s]".format(self.sim_time))
        print("  Controller: {} [s]".format(self.ctrl_time))
        print("  Sensors: {} [s]".format(self.sensors_time))
        print("  Logging: {} [s]".format(self.log_time))
        print("  Camera: {} [s]".format(self.camera_time))
        print("  Wait real-time: {} [s]".format(self.waitrt_time))
        print("  Sum: {} [s]".format(
            self.plugin_time
            + self.sim_time
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

    def spawn(self):
        """Spawn"""
        for element in self.elements():
            element.spawn()
        self.logger = ExperimentLogger(
            self.animat.model,
            self.n_iterations
        )

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

    def __init__(self, timestep, n_iterations, **kwargs):
        super(SalamanderExperiment, self).__init__(
            animat=Salamander(
                kwargs.pop("animat_options", ModelOptions()),
                timestep
            ),
            arena=FlooredArena(),
            timestep=timestep,
            n_iterations=n_iterations
        )


class Simulation:
    """Simulation"""

    def __init__(self, simulation_options, animat_options):
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
        self.experiment = SalamanderExperiment(
            self.timestep,
            len(self.times),
            animat_options=self.animat_options
        )
        self.experiment.spawn()
        self.animat = self.experiment.animat
        self.plane = self.experiment.arena.floor.identity
        self.animat.model.leg_collisions(self.plane.identity, activate=False)
        self.animat.model.print_dynamics_info()

        # Interface
        self.interface = Interfaces()
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
                size=len(self.times)//25,
                rotating_camera=self.sim_options.rotating_camera,
                top_camera=self.sim_options.top_camera
            )

        # Simulation
        self.profile = SimulationProfiler(self.sim_options.duration)
        self.forces_torques = np.zeros([len(self.times), 2, 10, 3])
        self.sim_step = 0

        # Simulation state
        self.init_state = pybullet.saveState()
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
        while self.sim_step < len(self.times):
            if not self.sim_options.headless:
                if not self.sim_step % 100:
                    self.interface.user_params.update()
                    keys = pybullet.getKeyboardEvents()
                    if ord("q") in keys:
                        break
                if not(self.sim_step % 10000) and self.sim_step > 0:
                    pybullet.restoreState(self.init_state)
                if not self.interface.user_params.play.value:
                    time.sleep(0.5)
            tic_loop = time.time()
            self.loop()
            loop_time += time.time() - tic_loop
        print("Loop time: {} [s]".format(loop_time))
        self.toc = time.time()
        self.times_simulated = self.times[:self.sim_step]

    def loop(self):
        """Simulation loop"""
        self.tic_rt = time.time()
        self.sim_time = self.timestep*self.sim_step
        # Control animat
        self.animat_control()
        # Physics
        self.tic_sim = time.time()
        pybullet.stepSimulation()
        self.sim_step += 1
        self.toc_sim = time.time()
        self.profile.sim_time += self.toc_sim - self.tic_sim
        # Animat logging
        self.animat_logging()
        # Camera
        tic_camera = time.time()
        if not self.sim_options.headless:
            if self.sim_options.record and not self.sim_step % 25:
                self.camera_record.record(self.sim_step//25-1)
            # User camera
            if (
                    not self.sim_step % self.interface.camera_skips
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
                time_control=self.time_control
            )
        self.profile.waitrt_time = time.time() - tic_rt

    def postprocess(self):
        """Plot after simulation"""
        # Plot
        self.experiment.logger.plot_all(self.times_simulated)
        plt.show()

        # Record video
        if self.sim_options.record and not self.sim_options.headless:
            self.camera_record.save("video.avi")

    def end(self):
        """Terminate simulation"""
        # Simulation information
        self.profile.sim_duration = self.timestep*self.sim_step
        self.profile.sim_time = self.toc - self.tic
        self.profile.print_times()

        # Disconnect from simulation
        pybullet.disconnect()


class SalamanderSimulation(Simulation):
    """Salamander simulation"""

    def __init__(self, simulation_options, animat_options):
        super(SalamanderSimulation, self).__init__(
            simulation_options=simulation_options,
            animat_options=animat_options
        )

    def animat_control(self):
        """Control animat"""
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
        # Swimming
        if self.animat_options.gait == "swimming":
            self.forces_torques[self.sim_step] = viscous_swimming(
                self.animat.identity,
                self.animat.links
            )
        # Time plugins
        self.time_plugin = time.time() - self.tic_rt
        self.profile.plugin_time += self.time_plugin
        # Control
        self.tic_control = time.time()
        self.animat.model.controller.control()
        self.time_control = time.time() - self.tic_control
        self.profile.ctrl_time += self.time_control

    def animat_logging(self):
        """Animat logging"""
        # Contacts during walking
        tic_sensors = time.time()
        self.animat.model.sensors.update(
            identity=self.animat.identity,
            links=[self.animat.links[foot] for foot in self.animat.model.feet],
            joints=[
                self.animat.joints[joint]
                for joint in self.animat.model.sensors.joints_sensors
            ],
            plane=self.plane.identity
        )
        # Commands
        self.animat.model.motors.update(
            identity=self.animat.identity,
            joints_body=[
                self.animat.joints[joint]
                for joint in self.animat.model.motors.joints_commanded_body
            ],
            joints_legs=[
                self.animat.joints[joint]
                for joint in self.animat.model.motors.joints_commanded_legs
            ]
        )
        time_sensors = time.time() - tic_sensors
        self.profile.sensors_time += time_sensors
        tic_log = time.time()
        self.experiment.logger.update(self.sim_step-1)
        time_log = time.time() - tic_log
        self.profile.log_time += time_log


def run_simon():
    """Run Simon's experiment"""
    pass


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
    sim.postprocess()
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
