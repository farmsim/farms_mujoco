#!/usr/bin/env python3
"""Salamander - Test evolution"""

import argparse

from salamander_evolution.problems import ProblemWalkingFrequency
from salamander_evolution.evolution import evolve


def parse_args():
    """ Parse arguments """
    parser = argparse.ArgumentParser(
        description='Test salamander evolution'
    )
    parser.add_argument(
        "-p", '--population',
        type=int,
        dest='population',
        default=5,
        help='Number of individuals in population'
    )
    parser.add_argument(
        "-g", '--generations',
        type=int,
        dest='generations',
        default=3,
        help='Number of generations'
    )
    parser.add_argument(
        "-a", '--algorithm',
        type=str,
        dest='algorithm',
        default="cmaes",
        help='Algorithm to use'
    )
    return parser.parse_args()


def main():
    """Main"""
    args = parse_args()
    problem = ProblemWalkingFrequency(link_name="body_link_0")
    evolve(
        problem=problem,
        algorithm=args.algorithm,
        n_population=args.population,
        n_generations=args.generations
    )


if __name__ == '__main__':
    main()
