#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import time
import matplotlib.pyplot as plt
from farms_bullet.experiments.salamander.simulation import main as run_sim
from farms_bullet.experiments.salamander.animat_options import SalamanderOptions
from farms_bullet.simulations.simulation_options import SimulationOptions


def main():
    """Main"""
    animat_options = SalamanderOptions(
        show_hydrodynamics=False
    )
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
    TIC = time.time()
    # main()
    profile()
    # pycall()
    print("Total simulation time: {} [s]".format(time.time() - TIC))
