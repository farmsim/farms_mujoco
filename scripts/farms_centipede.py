#!/usr/bin/env python3
"""Run centipede simulation with bullet"""

import time
import matplotlib.pyplot as plt
from farms_bullet.experiments.centipede.simulation import main as run_sim
from farms_bullet.animats.amphibious.animat_options import AmphibiousOptions
from farms_bullet.simulations.simulation_options import SimulationOptions


def main():
    """Main"""
    # Animat options
    animat_options = AmphibiousOptions(
        # collect_gps=True,
        # show_hydrodynamics=True,
        scale=1
    )
    # animat_options.control.drives.forward = 4
    # Simulation options
    simulation_options = SimulationOptions.with_clargs()
    simulation_options.units.meters = 1
    simulation_options.units.seconds = 1000
    simulation_options.units.kilograms = 1
    # Run simulation
    run_sim(
        simulation_options=simulation_options,
        animat_options=animat_options
    )
    # Show results
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
