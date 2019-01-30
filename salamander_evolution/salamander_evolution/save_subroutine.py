import datetime
import os
import imp
from initialization_subroutine import *
import initialization_subroutine
imp.reload(initialization_subroutine)

def save_gen(interval_observer, save_config, islands,
            gen, destination_dir_ADN, destination_dir_fit):
    """saving the whole population at generation i if save is ON
    saving the decision vector at the path save_ADN/.../topology...
    saving the objective function at save_fitness/.../topology..."""
    if interval_observer == save_config['interval'] and save_config['save_on'] == 1:
        x_isl_pop = [isl.get_population().get_x() for isl in islands]
        decision_vector_save = np.array(x_isl_pop)
        f_isl_pop = [isl.get_population().get_f() for isl in islands]
        fitness_save = np.array(f_isl_pop)

        np.savez(destination_dir_ADN + "/generation_{}".format(gen), decision_vector_save)
        np.savez(destination_dir_fit + "/generation_{}".format(gen), fitness_save)
        interval_observer = 0
    return interval_observer


def save_seed(gen, islands, destination_dir_ADN ):
    """save the seed of the whole islands in order to reconstruct the population
    if needed, take as input the generation and the islands pygmo model"""
    if gen == 0:
        print("seed saved")
        isl_seed = []
        isl_seed.append([isl.get_population().get_seed() for isl in islands])
        isl_seed = np.array(isl_seed)
        np.savez(destination_dir_ADN + "/seed_config", isl_seed)


def save_evol_config(evol_config,isl_topo):
    """save the config file of the evolution in a directory (path can be specified)"""
    now = datetime.datetime.now()
    current_dir = os.getcwd()
    destination_dir = os.path.join(current_dir, r'save_ADN/month_{}/day_{}/{}/topology_{}/'.format(
                                                                                    now.strftime("%m"),
                                                                                    now.strftime("%d"),
                                                                                    now.strftime("%Hh"),
                                                                                    isl_topo))
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    yaml_dump(destination_dir + "my_config.yaml", evol_config)


def write_time_fit_data(migration_config, min_fitness, evol_time):
    """Save the value in a csv in order to estimate the time of computation"""
    import csv

    for key, value in migration_config['scheme'].items():
        if value == True:
            scheme = key

    with open('data_time_fit_{}.csv'.format(scheme), mode='a') as data_fit_time:
        data_write = csv.writer(data_fit_time, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        data_write.writerow([np.round(min_fitness, decimals=0), np.round(evol_time, decimals=1)])