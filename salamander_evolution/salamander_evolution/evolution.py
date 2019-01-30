"""Evolution"""

import pygmo as pg


def evolve(problem, algorithm, n_population, n_generations):
    """Evolver interface"""
    # if algorithm == "cmaes":
    #     algo = pg.algorithm(pg.cmaes(gen=n_generations, force_bounds=True))
    # else:
    #     raise Exception("Salamander_evolution: Unrecognised algorithm")
    algo = pg.algorithm(pg.cmaes(gen=n_generations, force_bounds=True))
    pop = pg.population(pg.problem(problem), n_population)
    print("Running evolution")
    return algo.evolve(pop)
