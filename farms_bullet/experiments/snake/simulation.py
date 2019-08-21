"""Snake simulation"""

import time
import numpy as np
import pybullet
from ...simulations.simulation import Simulation, SimulationElements
from ...simulations.simulation_options import SimulationOptions
from ...arenas.arena import FlooredArena
from ...interface.interface import Interfaces
from ...simulations.simulator import real_time_handing
from ...sensors.logging import SensorsLogger

from .animat import Snake
from ..salamander.animat_options import SalamanderOptions


class SnakeSimulation(Simulation):
    """Snake simulation"""

    def __init__(self, simulation_options, animat_options, **kwargs):
        super(SnakeSimulation, self).__init__(
            elements=SimulationElements(
                animat=Snake(
                    animat_options,
                    simulation_options.timestep,
                    simulation_options.n_iterations,
                    simulation_options.units
                ),
                arena=kwargs.pop("arena", FlooredArena())
            ),
            options=simulation_options
        )
        # Logging
        self.logger = SensorsLogger(self.elements.animat.sensors)
        # Interface
        self.interface = Interfaces(int(10*1e-3/simulation_options.timestep))
        if not self.options.headless:
            self.interface.init_camera(
                target_identity=(
                    self.elements.animat.identity
                    if not self.options.free_camera
                    else None
                ),
                timestep=self.options.timestep,
                rotating_camera=self.options.rotating_camera,
                top_camera=self.options.top_camera
            )
            self.interface.init_debug(animat_options=self.elements.animat.options)

        if self.options.record and not self.options.headless:
            skips = int(2e-2/simulation_options.timestep)  # 50 fps
            self.interface.init_video(
                target_identity=self.elements.animat.identity,
                simulation_options=simulation_options,
                fps=1./(skips*simulation_options.timestep),
                pitch=-45,
                yaw=0,
                skips=skips,
                motion_filter=2*skips*simulation_options.timestep,
                distance=1,
                rotating_camera=self.options.rotating_camera,
                top_camera=self.options.top_camera
            )
        # Real-time handling
        self.tic_rt = np.zeros(2)
        # Simulation state
        self.simulation_state = None
        self.save()

    def pre_step(self, sim_step):
        """New step"""
        play = True
        # if not(sim_step % 10000) and sim_step > 0:
        #     pybullet.restoreState(self.simulation_state)
        #     state = self.elements.animat.data.state
        #     state.array[self.elements.animat.data.iteration] = (
        #         state.default_initial_state()
        #     )
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
        # Interface
        if not self.options.headless:
            if self.elements.animat.options.transition:
                self.interface.user_params.drive_speed.value = (
                    1+4*sim_step/self.options.n_iterations
                )
                self.interface.user_params.drive_speed.changed = True
            self.animat_interface()
        # Animat sensors
        self.elements.animat.sensors.update(sim_step)
        if sim_step < self.options.n_iterations-1:
            # Plugins
            if self.elements.animat.options.control.drives.forward > 3:
                # Swimming
                self.elements.animat.animat_swimming_physics(sim_step)
            if self.elements.animat.options.show_hydrodynamics:
                self.elements.animat.draw_hydrodynamics(sim_step)
            # Control animat
            self.elements.animat.controller.control()
            # Physics step
            pybullet.stepSimulation()
            sim_step += 1
            # Camera
            if not self.options.headless:
                if self.options.record:
                    self.interface.video.record(sim_step)
                # User camera
                self.interface.camera.update()
            # Real-time
            self.tic_rt[1] = time.time()
            if (
                    not self.options.fast
                    and self.interface.user_params.rtl.value < 2.99
            ):
                real_time_handing(
                    self.options.timestep,
                    self.tic_rt,
                    rtl=self.interface.user_params.rtl.value
                )

    def animat_interface(self):
        """Animat interface"""
        # Camera zoom
        if self.interface.user_params.zoom.changed:
            self.interface.camera.set_zoom(
                self.interface.user_params.zoom.value
            )
        # Body offset
        if self.interface.user_params.body_offset.changed:
            self.elements.animat.options.control.network.joints.body_offsets = (
                self.interface.user_params.body_offset.value
            )
            self.elements.animat.controller.network.update(
                self.elements.animat.options
            )
            self.interface.user_params.body_offset.changed = False
        # Drives
        if self.interface.user_params.drive_speed.changed:
            self.elements.animat.options.control.drives.forward = (
                self.interface.user_params.drive_speed.value
            )
            self.elements.animat.controller.network.update(
                self.elements.animat.options
            )
            if self.elements.animat.options.control.drives.forward > 3:
                pybullet.setGravity(0, 0, -0.01*self.options.units.gravity)
            else:
                pybullet.setGravity(0, 0, -9.81*self.options.units.gravity)
            self.interface.user_params.drive_speed.changed = False
        # Turning
        if self.interface.user_params.drive_turn.changed:
            self.elements.animat.options.control.drives.turning = (
                self.interface.user_params.drive_turn.value
            )
            self.elements.animat.controller.network.update(
                self.elements.animat.options
            )
            self.interface.user_params.drive_turn.changed = False


def main(simulation_options=None, animat_options=None):
    """Main"""

    # Parse command line arguments
    if not simulation_options:
        simulation_options = SimulationOptions.with_clargs()
    if not animat_options:
        animat_options = SalamanderOptions()
        animat_options.morphology.n_joints_body = 12
        animat_options.morphology.n_dof_legs = 0
        animat_options.morphology.n_legs = 0

    # Setup simulation
    print("Creating simulation")
    sim = SnakeSimulation(
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
