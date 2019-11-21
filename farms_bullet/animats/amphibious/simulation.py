"""Amphibious simulation"""

import time
import numpy as np
import pybullet

from ...simulations.simulation import Simulation, SimulationElements
from ...simulations.simulation_options import SimulationOptions
from ...arenas.arena import FlooredArena, ArenaWater
from ...interface.interface import Interfaces
from ...simulations.simulator import real_time_handing

from .animat import Amphibious
from .animat_options import AmphibiousOptions


class AmphibiousSimulation(Simulation):
    """Amphibious simulation"""

    def __init__(self, simulation_options, animat, **kwargs):
        if "arena" not in kwargs:
            arena = (
                ArenaWater(
                    ramp_angle=-10,
                    units=animat.units
                )
                if simulation_options.arena == "water"
                else FlooredArena()
            )
        super(AmphibiousSimulation, self).__init__(
            elements=SimulationElements(
                animat=animat,
                arena=kwargs.pop("arena", arena)
            ),
            options=simulation_options
        )
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
                pitch=simulation_options.video_pitch,
                yaw=simulation_options.video_yaw,
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

    @property
    def animat(self):
        """Salamander animat"""
        return self.elements.animat

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

            # Drive changes depending on simulation time
            if self.elements.animat.options.transition:
                self.interface.user_params.drive_speed.value = (
                    1+4*sim_step/self.options.n_iterations
                )
                self.interface.user_params.drive_speed.changed = True

            # # Switch drive based on position
            # distance = self.elements.animat.data.sensors.gps.com_position(
            #     iteration=sim_step-1 if sim_step else 0,
            #     link_i=0
            # )[0]
            # swim_distance = 3
            # value = self.interface.user_params.drive_speed.value
            # if distance < -swim_distance:
            #     self.interface.user_params.drive_speed.value = 4 - (
            #         0.05*(swim_distance+distance)
            #     )
            #     if self.interface.user_params.drive_speed.value != value:
            #         self.interface.user_params.drive_speed.changed = True
            # else:
            #     if self.interface.user_params.drive_speed.value != value:
            #         self.interface.user_params.drive_speed.changed = True

            # Update interface
            self.animat_interface()

        # Animat sensors
        self.elements.animat.sensors.update(sim_step)

        # Physics step
        if sim_step < self.options.n_iterations-1:
            physics_options = self.elements.animat.options.physics
            # Swimming
            if physics_options.viscous or physics_options.sph:
                water_surface = (
                    np.inf
                    if physics_options.sph or not physics_options.water_surface
                    else self.elements.arena.water_surface
                )
                if physics_options.viscous:
                    self.elements.animat.viscous_swimming_forces(
                        sim_step,
                        water_surface=water_surface,
                        coefficients=physics_options.viscous_coefficients,
                        buoyancy=physics_options.buoyancy,
                    )
                self.elements.animat.apply_swimming_forces(
                    sim_step,
                    water_surface=water_surface
                )
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
        if not self.options.headless:
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
            self.elements.animat.options.control.network.joints.set_body_offsets(
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
            # if self.elements.animat.options.control.drives.forward > 3:
            #     pybullet.setGravity(0, 0, -0.01*self.options.units.gravity)
            # else:
            #     pybullet.setGravity(0, 0, -9.81*self.options.units.gravity)
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
        animat_options = AmphibiousOptions()

    # Setup simulation
    print("Creating simulation")
    sim = AmphibiousSimulation(
        simulation_options=simulation_options,
        animat=Amphibious(
            animat_options,
            simulation_options.timestep,
            simulation_options.n_iterations,
            simulation_options.units
        )
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
    if simulation_options.log_path:
        np.save(
            simulation_options.log_path+"/hydrodynamics.npy",
            sim.elements.animat.data.sensors.hydrodynamics.array
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
