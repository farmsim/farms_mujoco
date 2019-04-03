#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

from farms_bullet.simulation import main as run_simulation
from farms_bullet.model_options import ModelOptions
from farms_bullet.simulation_options import SimulationOptions


def main():
    """Main"""
    model_options = ModelOptions()
    # model_options = ModelOptions(
    #     frequency=1.7,
    #     body_stand_amplitude=0.42
    # )
    # sim_options = SimulationOptions.with_clargs(duration=100)
    # run_simulation(sim_options=sim_options, model_options=model_options)
    run_simulation(model_options=model_options)


if __name__ == '__main__':
    main()
