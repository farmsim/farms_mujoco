"""Extensions"""

import os

import numpy as np
from imageio import imread

from farms_core import pylog
from farms_core.sensors.sensor_convention import sc
from farms_core.model.data import AnimatData
from farms_core.model.options import AnimatOptions, ArenaOptions
from farms_core.experiment.options import ExperimentOptions
from farms_core.model.extensions import AnimatExtension
from farms_mujoco.swimming.drag import SwimmingHandler
from farms_mujoco.simulation.mjcf import get_prefix
from farms_mujoco.swimming.drag import WaterPropertiesExtension


def water_velocity_from_maps(position, water_maps):
    """Water velocity from maps"""
    vel = np.zeros(3)
    if all(
            water_maps['pos_min'][i] < position[i] < water_maps['pos_max'][i]
            for i in range(2)
    ):
        vel[:2] = [
            water_maps[png][tuple(
                (
                    max(0, min(
                        water_maps[png].shape[index]-1,
                        round(water_maps[png].shape[index]*(
                            (
                                position[index]
                                - water_maps['pos_min'][index]
                            ) / (
                                water_maps['pos_max'][index]
                                - water_maps['pos_min'][index]
                            )
                        ))
                    ))
                )
                for index in range(2)
            )]
            for png_i, png in enumerate(['vel_x', 'vel_y'])
        ]
    # vel[1] *= -1
    return vel


def maps_surface_callback(surface_height):
    """Maps surface callback"""
    def surface_callback(t, x, y):  # pylint: disable=unused-argument
        """Surface"""
        return surface_height
    return surface_callback


def maps_density_callback(density):
    """Maps density callback"""
    def density_callback(t, x, y, z):  # pylint: disable=unused-argument
        """Density"""
        return density
    return density_callback


def maps_viscosity_callback(viscosity):
    """Maps viscosity callback"""
    def viscosity_callback(t, x, y, z):  # pylint: disable=unused-argument
        """Viscosity"""
        return viscosity
    return viscosity_callback


def maps_velocity_callback(water_maps):
    """Maps velocity callback"""
    def velocity_callback(t, x, y, z):  # pylint: disable=unused-argument
        """Velocity in global frame"""
        return water_velocity_from_maps(
            position=[x, y, z],
            water_maps=water_maps,
        )
    return velocity_callback


class SwimmingExtension(AnimatExtension):
    """Swimming extension"""

    def __init__(
            self,
            animat_i: int,
            animat_data: AnimatData,
            animat_options: AnimatOptions,
            arena_options: ArenaOptions,
            substep=True,
            water_properties=None,
    ):
        super().__init__(substep=substep)
        self.animat_i = animat_i
        self.animat_data = animat_data
        self.animat_options = animat_options
        self.arena_options = arena_options
        self._handler: SwimmingHandler = None
        self._water_properties = water_properties

        self.constant_velocity: bool = (
            len(arena_options.water.velocity) == 3
        )
        if not self.constant_velocity:
            water_velocity = arena_options.water.velocity
            water_maps = [
                os.path.expandvars(map_path)
                for map_path in arena_options.water.maps
            ]
            for i in range(2):
                assert os.path.isfile(water_maps[i]), (
                    f"{water_maps[i]=} is not pointing to an existing file"
                    f"\nNote: {arena_options.water.velocity=}"
                )
            pngs = [np.flipud(imread(water_maps[i])).T for i in range(2)]
            pngs_info = [np.iinfo(png.dtype) for png in pngs]
            vels = [
                (
                    png.astype(np.double) - info.min
                ) * (
                    water_velocity[png_i+3] - water_velocity[png_i+0]
                ) / (
                    info.max - info.min
                ) + water_velocity[png_i+0]
                for png_i, (png, info) in enumerate(zip(pngs, pngs_info))
            ]
            self.water_maps = {
                'pos_min': np.array(water_velocity[6:8]),
                'pos_max': np.array(water_velocity[8:10]),
                'vel_x': -vels[0],
                'vel_y': +vels[1],
            }
            pylog.debug(
                "Water velocities loaded: %s"
                "\nVelX: Min=%s [m/s] Max=%s [m/s] (%s/%s / %s/%s)"
                "\nVelY: Min=%s [m/s] Max=%s [m/s] (%s/%s / %s/%s)"
                "\nWater velocity: %s",
                water_maps,
                np.min(vels[0]), np.max(vels[0]),
                pngs[0].min(), pngs_info[0].min,
                pngs[0].max(), pngs_info[0].max,
                np.min(vels[1]), np.max(vels[1]),
                pngs[1].min(), pngs_info[1].min,
                pngs[1].max(), pngs_info[1].max,
                water_velocity,
            )
            wtr_options = arena_options.water
            self._water_properties = WaterPropertiesExtension(
                surface=maps_surface_callback(float(wtr_options.height)),
                density=maps_density_callback(float(wtr_options.density)),
                viscosity=maps_viscosity_callback(float(wtr_options.viscosity)),
                velocity=maps_velocity_callback(self.water_maps),
            )

    @classmethod
    def from_options(
            cls,
            config: dict,
            experiment_options: ExperimentOptions,
            animat_i: int,
            animat_data: AnimatData,
            animat_options: AnimatOptions,
    ):
        """From options"""
        water_properties = None
        return cls(
            animat_i=animat_i,
            animat_data=animat_data,
            animat_options=animat_options,
            arena_options=experiment_options.arenas[0],
            water_properties=water_properties,
        )

    def initialize_episode(self, task, physics):
        """Initialize episode"""
        self._handler = SwimmingHandler(
            data=self.animat_data,
            animat_options=self.animat_options,
            arena_options=self.arena_options,
            units=task.units,
            physics=physics,
            water=self._water_properties,
            prefix=get_prefix(self.animat_i),
        )

    def before_step(self, task, action, physics):
        """Step hydrodynamics"""
        del action

        # Compute fluid forces
        self._handler.step(physics.time()/task.units.seconds, task.iteration)

        # Set fluid forces in physics engine
        indices = task.maps[self.animat_i]['sensors']['data2xfrc']
        physics.data.xfrc_applied[indices, :] = (
            self.animat_data.sensors.xfrc.array[
                task.iteration, :,
                sc.xfrc_force_x:sc.xfrc_torque_z+1,
            ]
        )
        for force_i, (rotation_mat, force_local) in enumerate(zip(
                physics.data.xmat[indices],
                physics.data.xfrc_applied[indices],
        )):
            physics.data.xfrc_applied[indices[force_i]] = (
                rotation_mat.reshape([3, 3])  # Local to global frame
                @ force_local.reshape([3, 2], order='F')
            ).flatten(order='F')
        physics.data.xfrc_applied[indices, :3] *= task.units.newtons
        physics.data.xfrc_applied[indices, 3:] *= task.units.torques
