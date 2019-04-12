#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

from farms_bullet.simulations.salamander import main as run_simulation
from farms_bullet.animats.model_options import ModelOptions
from farms_bullet.simulations.simulation_options import SimulationOptions


def main():
    """Main"""
    animat_options = ModelOptions()
    # animat_options = ModelOptions(
    #     frequency=1.7,
    #     body_stand_amplitude=0.42
    # )
    simulation_options = SimulationOptions.with_clargs(duration=100)
    run_simulation(
        simulation_options=simulation_options,
        animat_options=animat_options
    )


if __name__ == '__main__':
    main()
