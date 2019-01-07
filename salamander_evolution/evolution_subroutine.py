import importlib as imp
import time
import results_subroutine
import migration_subroutine
import save_subroutine
import numpy as np
import pygmo as pg
from pygmo import *
from migration_subroutine import *
from results_subroutine import *
from save_subroutine import *

imp.reload(results_subroutine)
imp.reload(migration_subroutine)
imp.reload(save_subroutine)


def evolution_cycle(udas, algo_params_config, prob, pop_config,
                    verbosity_config, save_config, destination_dir_fit, destination_dir_ADN, migration_config):
    """subroutine for the evolution with the different topologies and parameters of the config.yalm
    construct the asynchronous model with the number of corresponding islands
    call evolution_result = evolution_cycle(...)
    return evolution_result[0] = data_f
    return evolution_result[1] = islands"""

    prob = pg.problem(prob)
    print("Problem:\n{}".format(prob))

    if algo_params_config['verbosity'] == True:
            udas = [pg.algorithm(uda) for uda in udas]
            for m in np.arange(0, len(udas)):
                udas[m].set_verbosity(1)
    if 0:
        pop = pg.population(prob, size = pop_config['size'])
        islands = [pg.island(algo = uda, pop = pop, udi = pg.mp_island(use_pool=True)) for uda in udas]
    elif 1:
        islands = [pg.island(algo = uda,
                  prob = prob,
                  size = pop_config['size'],
                  udi = pg.mp_island(use_pool=True)) for uda in udas]
    print(islands)
    data_f = []
    interval_observer = 0
    init_obs = 0
    stats_observer = 0
    adjacency_matrix = build_adjacency(islands, migration_config)
    start_time = time.time()
    for gen in np.arange(0, algo_params_config['generation']):
        _ = [isl.evolve() for isl in islands]
        _ = [isl.wait() for isl in islands]


        data_f.append([isl.get_population().champion_f[0] for isl in islands])

        #print(data_f)
        print_generation(gen, islands, verbosity_config)

        #my_check = np.array(data_f)
        if np.remainder(gen, 100) == 0:
            print (time.time()-start_time)


        islands = migration(islands, udas, migration_config, adjacency_matrix, prob, gen)

        save_gen(interval_observer, save_config, islands,
                 gen, destination_dir_ADN, destination_dir_fit)

        save_seed(gen, islands, destination_dir_ADN)

        interval_observer += 1

    data_f = np.array(data_f)

    return data_f, islands
