"""Salamander simulation with pybullet"""

import time

import numpy as np
import matplotlib.pyplot as plt

import pybullet

from .parse_args import parse_args
from .plugins.swimming import viscous_swimming
from .simulator import init_engine, real_time_handing
from .debug import test_debug_info
from .arenas import create_scene
from .render import rendering
from .model import Model, SalamanderModel
from .interface import UserParameters
from .camera import UserCamera, CameraRecord
from. logging import ExperimentLogger
from .simulation_options import SimulationOptions


class Simulation:
    """Simulation"""

    def __init__(self, timestep, duration, options, gait="walking"):
        super(Simulation, self).__init__()
        # Initialise engine
        init_engine()
        rendering(0)

        # Parameters
        # gait = "standing"
        self.gait = gait
        self.frequency = 1
        # gait = "swimming"
        self.timestep = timestep
        self.times = np.arange(0, duration, self.timestep)

        # Initialise
        self.model, self.plane = self.init_simulation(gait=gait)
        self.init_experiment(options)
        rendering(1)

    def init_simulation(self, gait="walking"):
        """Initialise simulation"""
        # Physics
        self.init_physics(gait)

        # Spawn models
        model = SalamanderModel.spawn(self.timestep, gait)
        plane = Model.from_urdf(
            "plane.urdf",
            basePosition=[0, 0, -0.1]
        )
        return model, plane

    def init_experiment(self, options):
        """Initialise simulation"""

        # Simulation entities
        self.salamander, self.links, self.joints, self.plane = (
            self.get_entities()
        )

        # Remove leg collisions
        self.salamander.leg_collisions(self.plane, activate=False)

        # Model information
        self.salamander.print_dynamics_info()

        # Create scene
        add_obstacles = False
        if add_obstacles:
            create_scene(self.plane)

        # Camera
        self.camera_skips = 10
        self.camera = UserCamera(
            target_identity=self.salamander.identity,
            yaw=0,
            yaw_speed=360/10*self.camera_skips if options.rotating_camera else 0,
            pitch=-89 if options.top_camera else -45,
            distance=1,
            timestep=self.timestep
        )

        # Video recording
        if options.record:
            self.camera_record = CameraRecord(
                target_identity=self.salamander.identity,
                size=len(self.times)//25,
                fps=40,
                yaw=0,
                yaw_speed=360/10 if options.rotating_camera else 0,
                pitch=-89 if options.top_camera else -45,
                distance=1,
                timestep=self.timestep*25,
                motion_filter=1e-1
            )

        # User parameters
        self.user_params = UserParameters(self.gait, self.frequency)

        # Debug info
        test_debug_info()

        # Simulation time
        self.tot_plugin_time = 0
        self.tot_sim_time = 0
        self.tot_ctrl_time = 0
        self.tot_sensors_time = 0
        self.tot_log_time = 0
        self.tot_camera_time = 0
        self.tot_waitrt_time = 0
        self.forces_torques = np.zeros([len(self.times), 2, 10, 3])
        self.sim_step = 0

        # Final setup
        self.experiment_logger = ExperimentLogger(
            self.salamander,
            len(self.times)
        )
        self.init_state = pybullet.saveState()
        rendering(1)

    def get_entities(self):
        """Get simulation entities"""
        return (
            self.model,
            self.model.links,
            self.model.joints,
            self.plane.identity
        )

    def init_physics(self, gait="walking"):
        """Initialise physics"""
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

    def run(self, options):
        """Run simulation"""
        # Run simulation
        self.tic = time.time()
        loop_time = 0
        while self.sim_step < len(self.times):
            if not(self.sim_step % 100):
                self.user_params.update()
                keys = pybullet.getKeyboardEvents()
                if ord("q") in keys:
                    break
            if not(self.sim_step % 10000) and self.sim_step > 0:
                pybullet.restoreState(self.init_state)
            if not self.user_params.play.value:
                time.sleep(0.5)
            else:
                tic_loop = time.time()
                self.loop(options)
                loop_time += time.time() - tic_loop
        print("Loop time: {} [s]".format(loop_time))
        self.toc = time.time()
        self.times_simulated = self.times[:self.sim_step]

    def loop(self, options):
        """Simulation loop"""
        self.tic_rt = time.time()
        self.sim_time = self.timestep*self.sim_step
        # Control
        if self.user_params.gait.changed:
            self.gait = self.user_params.gait.value
            self.model.controller.update_gait(
                self.gait,
                self.joints,
                self.timestep
            )
            pybullet.setGravity(
                0, 0, -1e-2 if self.gait == "swimming" else -9.81
            )
            self.user_params.gait.changed = False
        if self.user_params.frequency.changed:
            self.model.controller.update_frequency(
                self.user_params.frequency.value
            )
            self.user_params.frequency.changed = False
        if self.user_params.body_offset.changed:
            self.model.controller.update_body_offset(
                self.user_params.body_offset.value
            )
            self.user_params.body_offset.changed = False
        # Swimming
        if self.gait == "swimming":
            self.forces_torques[self.sim_step] = viscous_swimming(
                self.salamander.identity,
                self.links
            )
        # Time plugins
        self.time_plugin = time.time() - self.tic_rt
        self.tot_plugin_time += self.time_plugin
        # Control
        self.tic_control = time.time()
        self.model.controller.control()
        self.time_control = time.time() - self.tic_control
        self.tot_ctrl_time += self.time_control
        # Physics
        self.tic_sim = time.time()
        pybullet.stepSimulation()
        self.sim_step += 1
        self.toc_sim = time.time()
        self.tot_sim_time += self.toc_sim - self.tic_sim
        # Contacts during walking
        tic_sensors = time.time()
        self.salamander.sensors.update(
            identity=self.salamander.identity,
            links=[self.links[foot] for foot in self.salamander.feet],
            joints=[
                self.joints[joint]
                for joint in self.salamander.sensors.joints_sensors
            ],
            plane=self.plane
        )
        # Commands
        self.salamander.motors.update(
            identity=self.salamander.identity,
            joints_body=[
                self.joints[joint]
                for joint in self.salamander.motors.joints_commanded_body
            ],
            joints_legs=[
                self.joints[joint]
                for joint in self.salamander.motors.joints_commanded_legs
            ]
        )
        time_sensors = time.time() - tic_sensors
        self.tot_sensors_time += time_sensors
        tic_log = time.time()
        self.experiment_logger.update(self.sim_step-1)
        time_log = time.time() - tic_log
        self.tot_log_time += time_log
        # Camera
        tic_camera = time.time()
        if options.record and not self.sim_step % 25:
            self.camera_record.record(self.sim_step//25-1)
        # User camera
        if not self.sim_step % self.camera_skips and not options.free_camera:
            self.camera.update()
        self.tot_camera_time += time.time() - tic_camera
        # Real-time
        self.toc_rt = time.time()
        tic_rt = time.time()
        if not options.fast and self.user_params.rtl.value < 3:
            real_time_handing(
                self.timestep, self.tic_rt, self.toc_rt,
                rtl=self.user_params.rtl.value,
                time_plugin=self.time_plugin,
                time_sim=self.toc_sim-self.tic_sim,
                time_control=self.time_control
            )
        self.tot_waitrt_time = time.time() - tic_rt

    def end(self, options):
        """Terminate simulation"""
        # Simulation information
        self.sim_time = self.timestep*self.sim_step
        print("Time to simulate {} [s]: {} [s]".format(
            self.sim_time,
            self.toc-self.tic,
        ))
        print("  Plugin: {} [s]".format(self.tot_plugin_time))
        print("  Bullet physics: {} [s]".format(self.tot_sim_time))
        print("  Controller: {} [s]".format(self.tot_ctrl_time))
        print("  Sensors: {} [s]".format(self.tot_sensors_time))
        print("  Logging: {} [s]".format(self.tot_log_time))
        print("  Camera: {} [s]".format(self.tot_camera_time))
        print("  Wait real-time: {} [s]".format(self.tot_waitrt_time))
        print("  Sum: {} [s]".format(
            self.tot_plugin_time
            + self.tot_sim_time
            + self.tot_ctrl_time
            + self.tot_sensors_time
            + self.tot_log_time
            + self.tot_camera_time
            + self.tot_waitrt_time
        ))

        # Disconnect from simulation
        pybullet.disconnect()

        # Plot
        self.experiment_logger.plot_all(self.times_simulated)
        plt.show()

        # Record video
        if options.record:
            self.camera_record.save("video.avi")


def main():
    """Main"""

    # Parse command line arguments
    options = SimulationOptions.with_clargs()

    # Setup simulation
    sim = Simulation(
        timestep=1e-3,
        duration=10,
        options=options,
        gait="walking"
    )

    # Run simulation
    sim.run(options)

    # Show results
    sim.end(options)


def main_parallel():
    """Simulation with multiprocessing"""
    from multiprocessing import Pool

    # Parse command line arguments
    options = SimulationOptions.with_clargs()

    # Create Pool
    pool = Pool(2)

    # Run simulation
    pool.map(main, [options, options])
    print("Done")


if __name__ == '__main__':
    # main_parallel()
    main()
