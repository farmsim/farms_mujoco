import numpy as np
import matplotlib.pyplot as plt

def config_print(isl_topo, algo_params_config, isl_config, pop_config, name_algo):
    """print the configurations for the evolution"""
    if isl_topo == 'single':
        print ("algorithm >> {}".format(algo_params_config['name_A']))
    else :
        j = 1
        for i in name_algo:
            print ("algorithm_{} >> {}".format(j, i))
            j += 1

    print ("island topology >> {}".format(isl_topo))
    print ("island number >> {}".format(isl_config['number']))
    print ("generation >> {}".format(algo_params_config['generation']))
    print ("Population size >> {}".format(pop_config['size']))


def print_generation(gen, islands, verbosity_config):
    if 0 == np.remainder(gen, verbosity_config['print_interval']):
        print('generation: {}, fitness: {}'.format(gen,
                                                  min([isl.get_population().champion_f[0] for isl in islands])))

def print_res(data_f, name_algo, algo_params_config):
    min_fitness = min(np.amin(data_f, axis=0))
    max_fitness = max(data_f[algo_params_config['generation'] - 1, :])
    best_algo = np.where(data_f == min_fitness)
    worst_algo = np.where(data_f == max_fitness)
    print("minima = {} >> algorith: {}".format(min_fitness,
                                               name_algo[np.remainder(min(best_algo[1]),
                                                                 len(name_algo))]))
    print("maxima = {} >> algorithm: {}".format(max_fitness,
                                                name_algo[np.remainder(min(worst_algo[1]),
                                                                  len(name_algo))]))

    return min_fitness, max_fitness


def plot_res(data_f, name_algo, now):
    plt.figure('Evolution Summary')
    for k in np.arange(0, len(name_algo)):
        plt.plot(data_f[:, k], label='Islands {}'.format(name_algo[k]))
    plt.minorticks_on()
    plt.legend()
    plt.grid(b=True, which='major', color='k', linestyle='-', linewidth='0.5')
    plt.grid(b=True, which='minor', color='k', linestyle=':', linewidth='0.25')
    plt.xlabel('generation [i]')
    plt.ylabel('objective function')
    plt.show(block=False)
    #saving the figure in figure directory
    plt.savefig('evolution_graphs/month_{}/day_{}/Evolution{}.pdf'.format(now.strftime("%m"),
                                                                     now.strftime("%d"),
                                                                     now.strftime("%Y_%m_%d_%Hh_%Mmin_%Ssec")),
                bbox_inches='tight')


def plot_inter(data_f, algo_params_config, fig_config, name_algo, udas, isl_topo, now, name):
    """plot the result with an confidence interval with a normal distribution
    the result of the evolution are stored in the matrix data_f, the algorithms
    are stored in the vector udas"""
    plt.figure('Confidence interval')
    plt.title('Metaheuristic optimization')
    if isl_topo == 'single':
        mean_spec_algo = np.mean(data_f, axis=1)
        std_spec_algo = np.std(data_f, axis=1)
        plt.plot(np.arange(0, algo_params_config['generation']),
                 mean_spec_algo,
                 lw=1,
                 alpha=1,
                 label='Fit')
        plt.fill_between(np.arange(0, algo_params_config['generation']),
                          mean_spec_algo - fig_config['confidence_interval'] * std_spec_algo,
                          mean_spec_algo + fig_config['confidence_interval'] * std_spec_algo,
                          alpha=0.4,
                          label='95% CI')

    if isl_topo == 'mixed' or isl_topo == 'full':
        mean_spec_algo = []
        std_spec_algo = []
        for m in np.arange(0, len(name)):
            mean_algo = data_f[:, np.arange(m,
                                            len(udas) + m,
                                            len(name))]
            mean_spec_algo.append(np.mean(mean_algo, axis=1))
            std_spec_algo.append(np.std(mean_algo, axis=1))

        mean_spec_algo = np.array(mean_spec_algo)
        std_spec_algo = np.array(std_spec_algo)
        for m in np.arange(0, len(name)):
            plt.plot(np.arange(0, algo_params_config['generation']),
                     mean_spec_algo[m, :],
                     lw=1,
                     alpha=1,
                     label='Fit')
            plt.fill_between(np.arange(0, algo_params_config['generation']),
                             mean_spec_algo[m, :] - fig_config['confidence_interval'] * std_spec_algo[m, :],
                             mean_spec_algo[m, :] + fig_config['confidence_interval'] * std_spec_algo[m, :],
                             alpha=0.4,
                             label='95% CI')

    plt.xlabel('Generation')
    plt.ylabel('Fitness function')
    plt.legend(name_algo[:len(name)])
    plt.show(block=False)
    plt.savefig('evolution_graphs/month_{}/day_{}/Evolution{}_conf.pdf'.format(now.strftime("%m"),
                        now.strftime("%d"),
                        now.strftime("%Y_%m_%d_%Hh_%Mmin_%Ssec")),
                        bbox_inches='tight')


    return mean_spec_algo, std_spec_algo


def plot_archi_topologies():
    data_1 = np.genfromtxt('data_time_fit_low_density.csv', delimiter=',',names=['density_fitness', 'density_time'])
    data_2 = np.genfromtxt('data_time_fit_full.csv', delimiter=',',names=['full_fitness', 'full_time'])
    data_3 = np.genfromtxt('data_time_fit_rand.csv', delimiter=',',names=['rand_fitness', 'rand_time'])
    data_4 = np.genfromtxt('data_time_fit_ring.csv', delimiter=',',names=['ring_fitness', 'ring_time'])
    plt.figure()

    density_xdata = data_1['density_time']
    full_xdata = data_2['full_time']
    rand_xdata = data_3['rand_time']
    ring_xdata = data_4['ring_time']

    density_ydata = data_1['density_fitness']
    full_ydata = data_2['full_fitness']
    rand_ydata = data_3['rand_fitness']
    ring_ydata = data_4['ring_fitness']

    id_sort_density = np.argsort(density_xdata)
    id_sort_full = np.argsort(full_xdata)
    id_sort_rand = np.argsort(rand_xdata)
    id_sort_ring = np.argsort(ring_xdata)


    plt.plot(density_xdata[id_sort_density], density_ydata[id_sort_density] , 'b.-', marker = '.', label='low density toplogy')
    plt.plot(full_xdata[id_sort_full], full_ydata[id_sort_full] , 'r.--', marker = '.', label='full topology')
    plt.plot(rand_xdata[id_sort_rand], rand_ydata[id_sort_rand] , 'm.--', marker = '.', label='rand topology')
    plt.plot(ring_xdata[id_sort_ring], ring_ydata[id_sort_ring] , 'k.--', marker = '.', label='ring topology')
    plt.xlim(-5,75)
    plt.xlabel('computational time [s]')
    plt.ylabel('objective function')
    plt.legend()

    plt.show(block=False)
