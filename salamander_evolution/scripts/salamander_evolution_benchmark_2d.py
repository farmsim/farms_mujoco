#!/usr/bin/env python3
"""Salamander - Test Pagmo2 evolution"""

import argparse

import pygmo as pg

import matplotlib.pyplot as plt

from salamander_evolution.viewer import AlgorithmViewer2D


def parse_args():
    """ Parse arguments """
    parser = argparse.ArgumentParser(description='Test evolution')
    # parser.add_argument(
    #     'model_names',
    #     type=str,
    #     nargs='+',
    #     help='Name of Gazebo model to simulate'
    # )
    parser.add_argument(
        "-s", '--save',
        action='store_true',
        dest='save',
        help='Save results'
    )
    args = parser.parse_args()
    return args


def main(n_threads=8):
    """Main"""

    args = parse_args()
    algorithms = []

    # Population without memory
    kwargs = {"seed": 0}
    algorithms.append([pg.de(gen=1, **kwargs) for _ in range(n_threads)])
    algorithms.append([pg.sea(gen=1, **kwargs) for _ in range(n_threads)])
    algorithms.append([pg.sga(gen=1, **kwargs) for _ in range(n_threads)])
    algorithms.append([pg.bee_colony(gen=1, **kwargs) for _ in range(n_threads)])
    # algorithm = pg.simulated_annealing()
    # algorithm = pg.ihs(gen=1, bw_min=1e-2, **kwargs)

    # Population with memory
    kwargs = {"memory": True, "seed": 0}
    algorithms.append([pg.pso(gen=1, **kwargs) for _ in range(n_threads)])
    algorithms.append([pg.sade(gen=1, **kwargs) for _ in range(n_threads)])
    algorithms.append([pg.de1220(gen=1, **kwargs) for _ in range(n_threads)])

    # Population with memory and bounds
    kwargs = {"memory": True, "seed": 0, "force_bounds": True}
    algorithms.append([pg.cmaes(gen=1, **kwargs) for _ in range(n_threads)])
    algorithms.append([pg.xnes(gen=1, **kwargs) for _ in range(n_threads)])

    # Multiobjective
    # algorithm = pg.nsga2(gen=1, **kwargs)
    # algorithm = pg.moead(gen=1, **kwargs)

    # Local
    # algorithm = pg.compass_search(max_fevals=100)
    # algorithm = pg.nlopt(solver="cobyla")
    # algorithm = pg.nlopt(solver="bobyqa")
    # algorithm = pg.nlopt(solver="neldermead")

    # Instantiate viewers
    viewers = [
        AlgorithmViewer2D(_algorithms, n_pop=10, n_gen=100)
        for _algorithms in algorithms
    ]
    # Run evolutions
    print("Running archipelago evolutions")
    for viewer in viewers:
        viewer.run_evolutions()
    # Animate
    for viewer in viewers:
        viewer.animate(write=args.save)
    # Save
    if not args.save:
        plt.show()


if __name__ == '__main__':
    main()
