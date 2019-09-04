"""Farms multiobjective optimisation for salamander"""

from farms_bullet.evolution.evolution import SalamanderEvolution, run_evolution


def main():
    """Main"""
    run_evolution(
        problem=SalamanderEvolution(),
        n_pop=40,
        n_gen=5
    )


if __name__ == '__main__':
    main()
