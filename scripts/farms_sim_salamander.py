#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

from farms_bullet.simulation import main as run_simulation
from farms_bullet.model_options import ModelOptions


def main():
    """Main"""
    model_options = ModelOptions()
    # model_options = ModelOptions(
    #     frequency=1.7,
    #     body_stand_amplitude=0.42
    # )
    run_simulation(model_options=model_options)


if __name__ == '__main__':
    main()
