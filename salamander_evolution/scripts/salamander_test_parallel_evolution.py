#!/usr/bin/env python3
"""Salamander - Test Pagmo2 evolution"""

import time
import argparse

import pygmo as pg

import matplotlib.pyplot as plt

from salamander_evolution.viewer import AlgorithmViewer2D


# def run_archipelago():
#     """ Run archipelago"""
#     algo = pg.algorithm(pg.cmaes(gen=3, force_bounds=True))
#     prob = pg.problem(QuadraticFunction())
#     archi = pg.archipelago(n=10, algo=algo, prob=prob, pop_size=5)
#     print("\nRUNNING EVOLUTION\n")
#     archi.evolve()
#     while(archi.status == pg.evolve_status.busy):
#         print("Status: {}".format(archi.status))
#         time.sleep(0.5)
#     print(archi.status)


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


def run_evolutions(viewer):
    """Run evolution"""
    viewer.run_evolutions()


def main():
    """Main"""

    args = parse_args()
    algorithms = []
    n_threads = 8

    # Population without memory
    kwargs = {"seed": 0}
    # algorithm = pg.de(gen=1, **kwargs)
    # algorithm = pg.de(gen=1, variant=1, **kwargs)
    # algorithm = pg.sea(gen=1, **kwargs)
    # algorithm = pg.sga(gen=1, **kwargs)
    # algorithm = pg.bee_colony(gen=1, **kwargs)
    # algorithm = pg.simulated_annealing()
    # algorithm = pg.ihs(gen=1, bw_min=1e-2, **kwargs)

    algorithms.append([pg.de(gen=1, **kwargs) for _ in range(n_threads)])
    algorithms.append([pg.sea(gen=1, **kwargs) for _ in range(n_threads)])
    algorithms.append([pg.sga(gen=1, **kwargs) for _ in range(n_threads)])
    algorithms.append([pg.bee_colony(gen=1, **kwargs) for _ in range(n_threads)])

    # Population with memory
    kwargs = {"memory": True, "seed": 0}
    # algorithm = pg.cmaes(gen=1, force_bounds=True, **kwargs)
    # algorithm = pg.xnes(gen=1, force_bounds=True, **kwargs)
    # algorithm = pg.pso(gen=1, **kwargs)
    # algorithm = pg.pso_gen(gen=1, **kwargs)
    # algorithm = pg.sade(gen=1, **kwargs)
    # algorithm = pg.sade(gen=1, variant=13, **kwargs)
    # algorithm = pg.sade(gen=1, xtol=1e0, ftol=1e0, **kwargs)
    # algorithm = pg.sade(gen=1, variant=11, variant_adptv=1, **kwargs)
    # algorithm = pg.sade(gen=1, variant=2, variant_adptv=2, **kwargs)
    # algorithm = pg.de1220(gen=1, **kwargs)

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

    # algorithms = []
    # kwargs = {"memory": True, "seed": 0}
    # algorithms.append([
    #     pg.sade(gen=1, variant=variant//2+1, **kwargs)
    #     for variant in range(8)
    # ])

    # Instantiate viewers
    viewers = [
        AlgorithmViewer2D(_algorithms, n_pop=10, n_gen=100)
        for _algorithms in algorithms
    ]
    print("Running islands")
    # Run evolutions
    for viewer in viewers:
        viewer.run_evolutions()
    # from multiprocessing import Pool
    # p = Pool(4)
    # print(p.map(run_evolutions, viewers))
    # Animate
    for viewer in viewers:
        viewer.animate(write=args.save)
    # Save
    if not args.save:
        plt.show()


if __name__ == '__main__':
    main()
