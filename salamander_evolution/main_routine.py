"""
Test for the toolbox of the evolution$
email: blaise.etter@epfl.ch
author copyright
"""



import time
import importlib as imp
import csv
import subprocess
import matplotlib
import numpy as np
import pygmo as pg
import networkx as nx
import matplotlib.pyplot as plt
from pygmo import *
from scipy import optimize
from scipy.optimize import curve_fit
from initialization_subroutine import *
from evolution_subroutine import *
from save_subroutine import *
from simulation_subroutine import *
from migration_subroutine import *
from results_subroutine import *
import initialization_subroutine
import save_subroutine
import simulation_subroutine
import migration_subroutine
import results_subroutine
import evolution_subroutine

imp.reload(initialization_subroutine)
imp.reload(save_subroutine)
imp.reload(simulation_subroutine)
imp.reload(migration_subroutine)
imp.reload(results_subroutine)
imp.reload(evolution_subroutine)
plt.close("all")

# file path for yaml file and importing the yalm file
filepath = "evol_config.yaml"
evol_config = yaml_loader(filepath)

# defining the different configuration in use
algo_config = evol_config['algorithm']
algo_params_config = evol_config['algorithm_params']
prob_config = evol_config['problem']
pop_config = evol_config['population']
model_config = evol_config['model']
isl_config = evol_config['island']
isl_topology = isl_config['topology']
save_config = evol_config['save']
fig_config = evol_config['figure']
migration_config = evol_config['migration']
font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}
matplotlib.rc('font', **font)
# verbosity configuration
verbosity_config = evol_config['verbosity']
# computing the vector name
name = list(filter(None, [algo_params_config['name_A'],
                     algo_params_config['name_B'],
                     algo_params_config['name_C'],
                     algo_params_config['name_D'],
                     algo_params_config['name_E'],
                     algo_params_config['name_F'],
                     algo_params_config['name_G']]))

# isl_topology(isl_topology)

for key, value in isl_topology.items():
    if value == True:
        isl_topo = key

# save the configuration file in the directory of saving
save_evol_config(evol_config, isl_topo)

# creating the directory for the save
now = datetime.datetime.now()
current_dir = os.getcwd()
destination_dir_ADN = os.path.join(current_dir,
                                   r'save_ADN/month_{}/day_{}/{}/topology_{}'.format(
                                       now.strftime("%m"),
                                       now.strftime("%d"),
                                       now.strftime("%Hh"),
                                       isl_topo))
destination_dir_fit = os.path.join(current_dir,
                                   r'save_fitness//month_{}/day_{}/topology_{}'.format(
                                       now.strftime("%m"),
                                       now.strftime("%d"),
                                       now.strftime("%Hh"),
                                       isl_topo))
destination_dir_fig = os.path.join(current_dir,
                                   r'evolution_graphs/month_{}/day_{}'.format(
                                       now.strftime("%m"),
                                       now.strftime("%d")))
if not os.path.exists(destination_dir_ADN):
    os.makedirs(destination_dir_ADN)
if not os.path.exists(destination_dir_fit):
    os.makedirs(destination_dir_fit)
if not os.path.exists(destination_dir_fig):
    os.makedirs(destination_dir_fig)

# computing the optimization problem to solve
#prob = pg.schwefel(1000)
prob = evol_problem(dim = 1,
                    link_name = "body_link_0",
                    path = ".gazebo/models/salamander_new")

# ===============================topo single=======================================
if isl_topo == 'single':
    name_algo = []
    udas = []
    for m in np.arange(0, isl_config['number']):

        if algo_params_config['name_A'] == 'de':
            de_params = algo_config['de']
            udas.append(init_de(de_params))

        elif algo_params_config['name_A'] == 'sga':
            sga_params = algo_config['sga']
            udas.append(init_sga(sga_params))

        elif algo_params_config['name_A'] == 'sade':
            sade_params = algo_config['sade']
            udas.append(init_sade(sade_params))

        elif algo_params_config['name_A'] == 'pso':
            pso_params = algo_config['pso']
            udas.append(init_pso(pso_params))

        elif algo_params_config['name_A'] == 'bee_colony':
            bee_colony_params = algo_config['bee_colony']
            udas.append(init_bee_colony(bee_colony_params))

        elif algo_params_config['name_A'] == 'de1220':
            de1220_params = algo_config['de1220']
            udas.append(init_de1220(de1220_params))

        elif algo_params_config['name_A'] == 'cmaes':
            cmaes_params = algo_config['cmaes']
            udas.append(init_cmaes(cmaes_params))

        elif algo_params_config['name_A'] == 'psogen':
            pso_gen_params = algo_config['psogen']
            udas.append(init_pso_gen(pso_gen_params))

        elif algo_params_config['name_A'] == 'xnes':
            xnes_params = algo_config['xnes']
            udas.append(init_xnes(xnes_params))

        elif algo_params_config['name_A'] == 'moead':
            moead_params = algo_config['moead']
            udas.append(init_moead(moead_params))

    name_algo = get_udas_name(udas)
    config_print(isl_topo, algo_params_config, isl_config, pop_config, name_algo)

    start_time = time.time()
    evol_res = evolution_cycle(udas, algo_params_config, prob, pop_config,
                               verbosity_config, save_config, destination_dir_fit, destination_dir_ADN,
                               migration_config)

    evol_time = time.time() - start_time
    print("Evolution_time %s [s]" % evol_time)

    data_f = evol_res[0]
    islands = evol_res[1]
    # migration_population = evol_res[2]

    data_plot = plot_inter(data_f, algo_params_config,
                           fig_config, name_algo,
                           udas, isl_topo, now, name)

    plot_res(data_f, name_algo, now)

    min_fitness, max_fitness = print_res(data_f, name_algo, algo_params_config)

    write_time_fit_data(migration_config, min_fitness, evol_time)


elif isl_topo == 'mixed' or isl_topo == 'full':

    udas = []

    print ("island topology >> {}".format(isl_topo))

    scale_factor = (isl_config['number'] - np.remainder(isl_config['number'], len(name))) / len(name)
    if scale_factor < 1:
        scale_factor = 1

    print ("island number >> {}".format(len(name) * scale_factor))
    print ("generation >> {}".format(algo_params_config['generation']))

    udas = []
    if isl_topo == "mixed":
        for m in np.arange(0, scale_factor):
            for iter_name in name:
                if iter_name == 'de':
                    de_params = algo_config['de']
                    udas.append(init_de(de_params))

                elif iter_name == 'sga':
                    sga_params = algo_config['sga']
                    udas.append(init_sga(sga_params))

                elif iter_name == 'sade':
                    sade_params = algo_config['sade']
                    udas.append(init_sade(sade_params))

                elif iter_name == 'pso':
                    pso_params = algo_config['pso']
                    udas.append(init_pso(pso_params))

                elif iter_name == 'bee_colony':
                    bee_colony_params = algo_config['bee_colony']
                    udas.append(init_bee_colony(bee_colony_params))

                elif iter_name == 'de1220':
                    de1220_params = algo_config['de1220']
                    udas.append(init_de1220(de1220_params))

                elif iter_name == 'cmaes':
                    cmaes_params = algo_config['cmaes']
                    udas.append(init_cmaes(cmaes_params))

                elif iter_name == 'psogen':
                    pso_gen_params = algo_config['psogen']
                    udas.append(init_pso_gen(pso_gen_params))

                elif iter_name == 'xnes':
                    xnes_params = algo_config['xnes']
                    udas.append(init_xnes(xnes_params))

                elif iter_name == 'moead':
                    moead_params = algo_config['moead']
                    udas.append(init_moead(moead_params))

        name_algo = get_udas_name(udas)
        config_print(isl_topo, algo_params_config, isl_config, pop_config, name_algo)

        comp_obj = 0
        for k in [10000]:#np.append(np.arange(100, 10000, 200), np.arange(10000, 30000, 1000)):
            algo_params_config['generation'] = k
            start_time = time.time()
            evol_res = evolution_cycle(udas, algo_params_config, prob, pop_config,
                                       verbosity_config, save_config, destination_dir_fit, destination_dir_ADN,
                                       migration_config)

            evol_time = time.time() - start_time
            print("Evolution_time %s [s]" % evol_time)


            data_f = evol_res[0]
            islands = evol_res[1]

            min_fitness, max_fitness = print_res(data_f, name_algo, algo_params_config)
            write_time_fit_data(migration_config, min_fitness, evol_time)

    elif isl_topo == "full":
        treshold = (isl_config['number'] - np.remainder(isl_config['number'], 10)) / 10
        if treshold < 1:
            treshold = 1
        for k in np.arange(0, treshold):
            de_params = algo_config['de']
            udas.append(init_de(de_params))

            sga_params = algo_config['sga']
            udas.append(init_sga(sga_params))

            sade_params = algo_config['sade']
            udas.append(init_sade(sade_params))

            pso_params = algo_config['pso']
            udas.append(init_pso(pso_params))

            bee_colony_params = algo_config['bee_colony']
            udas.append(init_bee_colony(bee_colony_params))

            de1220_params = algo_config['de1220']
            udas.append(init_de1220(de1220_params))

            cmaes_params = algo_config['cmaes']
            udas.append(init_cmaes(cmaes_params))

            psogen_params = algo_config['psogen']
            udas.append(init_pso_gen(psogen_params))

            xnes_params = algo_config['xnes']
            udas.append(init_xnes(xnes_params))

            moead_params = algo_config['moead']
            udas.append(init_moead(moead_params))

        name_algo = get_udas_name(udas)
        config_print(isl_topo, algo_params_config, isl_config, pop_config, name_algo)

        start_time = time.time()
        evol_res = evolution_cycle(udas, algo_params_config, prob, pop_config,
                                   verbosity_config, save_config, destination_dir_fit, destination_dir_ADN,
                                   migration_config)

        evol_time = time.time() - start_time
        print("Evolution_time %s [s]" % evol_time)
        data_f = evol_res[0]
        islands = evol_res[1]

    data_plot = plot_inter(data_f, algo_params_config,
                           fig_config, name_algo,
                           udas, isl_topo, now, name)

    plot_res(data_f, name_algo, now)

    min_fitness, max_fitness = print_res(data_f, name_algo, algo_params_config)

for key, value in migration_config['scheme'].items():
    if value == True:
        scheme = key

with open('data_time_fit_{}.csv'.format(scheme), mode='a') as data_fit_time:
    data_write = csv.writer(data_fit_time, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    data_write.writerow([np.round(min_fitness, decimals=0), np.round(evol_time, decimals=1)])

data_stats = np.genfromtxt('data_time_fit_{}.csv'.format(scheme), delimiter=',', names=['fitness', 'time'])
plt.figure()
plt.title('Objective function vs time')
xdata = data_stats['time']
ydata = data_stats['fitness']
id_sort = np.argsort(xdata)
plt.plot(xdata[id_sort], ydata[id_sort], 'b.-', marker='.', label='the data')
plt.xlim(-5, 100)
plt.xlabel('time [s]')
plt.ylabel('Objective function [-]')
plt.legend()

plt.show(block=False)