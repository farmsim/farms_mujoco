#!/usr/bin/env python3
"""Run snake simulation with bullet"""

import matplotlib.pyplot as plt
from farms_bullet.experiments.snake.simulation import main as run_sim
from farms_bullet.experiments.snake.animat_options import SnakeOptions
from farms_bullet.simulations.simulation_options import SimulationOptions


def main():
    """Main"""
    animat_options = SnakeOptions()
    # animat_options = SnakeOptions(
    #     frequency=1.7,
    #     body_stand_amplitude=0.42
    # )
    simulation_options = SimulationOptions.with_clargs()
    run_sim(
        simulation_options=simulation_options,
        animat_options=animat_options
    )
    plt.show()


def profile():
    """Profile with cProfile"""
    import cProfile
    import pstats
    cProfile.run("main()", "simulation.profile")
    pstat = pstats.Stats("simulation.profile")
    pstat.sort_stats('time').print_stats(30)
    pstat.sort_stats('cumtime').print_stats(30)


def pycall():
    """Profile with pycallgraph"""
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput
    with PyCallGraph(output=GraphvizOutput()):
        main()


if __name__ == '__main__':
    # main()
    profile()
    # pycall()
