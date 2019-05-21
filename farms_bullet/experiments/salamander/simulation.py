"""Salamander simulation"""

import time
import numpy as np
import pybullet
from ...simulations.simulation import Simulation, SimulationElements
from ...simulations.simulation_options import SimulationOptions
from .animat import Salamander
from ...animats.model_options import ModelOptions
from ...arenas.arena import FlooredArena
from ...interface.interface import Interfaces
from ...simulations.simulator import real_time_handing
from ...sensors.logging import SensorsLogger


class SalamanderSimulation(Simulation):
    """Salamander simulation"""

    def __init__(self, simulation_options, animat_options):
        super(SalamanderSimulation, self).__init__(
            elements=SimulationElements(
                animat=Salamander(
                    animat_options,
                    simulation_options.timestep,
                    simulation_options.n_iterations
                ),
                arena=FlooredArena()
            ),
            options=simulation_options
        )
        self.interface = Interfaces(int(10*1e-3/simulation_options.timestep))
        self.spawn()
        self.simulation_state = None
        self.tic_rt = np.zeros(2)
        self.save()

    def spawn(self):
        """Spawn"""
        # Elements
        self.elements.animat.add_sensors(self.elements.arena.floor.identity)
        self.logger = SensorsLogger(self.elements.animat.sensors)
        # Collisions
        self.elements.animat.model.leg_collisions(
            self.elements.arena.floor.identity,
            activate=False
        )
        self.elements.animat.model.print_dynamics_info()
        # Interface
        if not self.options.headless:
            self.interface.init_camera(
                target_identity=self.elements.animat.identity,
                timestep=self.options.timestep,
                rotating_camera=self.options.rotating_camera,
                top_camera=self.options.top_camera
            )
            self.interface.init_debug(animat_options=self.elements.animat.options)
        if self.options.record and not self.options.headless:
            self.interface.init_video(
                target_identity=self.elements.animat.identity,
                timestep=self.options.timestep*25,
                size=self.options.n_iterations//25,
                rotating_camera=self.options.rotating_camera,
                top_camera=self.options.top_camera
            )

    def pre_step(self, sim_step):
        """New step"""
        play = True
        if not(sim_step % 10000) and sim_step > 0:
            pybullet.restoreState(self.simulation_state)
            network = self.elements.animat.model.controller.network
            network.state.array[network.iteration] = (
                network.state.default_initial_state()
            )
        if not self.options.headless:
            play = self.interface.user_params.play.value
            if not sim_step % 100:
                self.interface.user_params.update()
            if not play:
                time.sleep(0.5)
                self.interface.user_params.update()
        return play

    def step(self, sim_step):
        """Simulation step"""
        self.tic_rt[0] = time.time()
        # Animat sensors
        self.elements.animat.sensors.update(sim_step)
        if sim_step < self.options.n_iterations-1:
            if not self.options.headless:
                self.animat_interface()
            # Plugins
            self.elements.animat.animat_physics()
            # if external_forces is not None:
            #     self.forces_torques[sim_step] = external_forces
            # Control animat
            self.elements.animat.animat_control()
            # Interface
            # Physics
            pybullet.stepSimulation()
            sim_step += 1
            # Camera
            if not self.options.headless:
                if self.options.record and not sim_step % 25:
                    self.elements.camera_record.record(sim_step//25-1)
                # User camera
                if (
                        not sim_step % self.interface.camera_skips
                        and not self.options.free_camera
                ):
                    self.interface.camera.update()
            # Real-time
            self.tic_rt[1] = time.time()
            if (
                    not self.options.fast
                    and self.interface.user_params.rtl.value < 3
            ):
                real_time_handing(
                    self.options.timestep,
                    self.tic_rt,
                    rtl=self.interface.user_params.rtl.value
                )

    def animat_interface(self):
        """Animat interface"""
        # Control
        if self.interface.user_params.gait.changed:
            self.elements.animat.options.gait = (
                self.interface.user_params.gait.value
            )
            self.elements.animat.model.controller.update_gait(
                self.elements.animat.options.gait,
                self.elements.animat.joints,
                self.options.timestep
            )
            if self.elements.animat.options.gait == "swimming":
                pybullet.setGravity(0, 0, -0.01)
            else:
                pybullet.setGravity(0, 0, -9.81)
            self.interface.user_params.gait.changed = False
        if self.interface.user_params.frequency.changed:
            network = self.elements.animat.model.controller.network
            network.parameters.oscillators.freqs = (
                self.interface.user_params.frequency.value
            )
            self.interface.user_params.frequency.changed = False
        if self.interface.user_params.body_offset.changed:
            network = self.elements.animat.model.controller.network
            network.parameters.joints.set_body_offset(
                self.interface.user_params.body_offset.value
            )
            self.interface.user_params.body_offset.changed = False
        if (
                self.interface.user_params.drive_speed.changed
                or self.interface.user_params.drive_turn.changed
        ):
            self.elements.animat.model.controller.network.update_drive(
                self.interface.user_params.drive_speed.value,
                self.interface.user_params.drive_turn.value
            )
            self.interface.user_params.drive_speed.changed = False
            self.interface.user_params.drive_turn.changed = False


def main(simulation_options=None, animat_options=None):
    """Main"""

    # Parse command line arguments
    if not simulation_options:
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

    # Analyse results
    print("Analysing simulation")
    sim.postprocess(
        iteration=sim.iteration,
        plot=simulation_options.plot,
        log_path=simulation_options.log_path,
        log_extension=simulation_options.log_extension,
        record=sim.options.record and not sim.options.headless
    )
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
