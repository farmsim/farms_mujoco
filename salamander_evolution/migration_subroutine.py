import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt
import networkx as nx

def islands_fevals(migration_population):
    """return the number of fevals of the whole islands model"""
    fevals = [migration_population[k].problem.get_fevals() for k in np.arange(0,len(migration_population))]

    return fevals


def migration(islands, udas, migration_config, adjacency_matrix, prob, gen):
    """migration population np.shape(#islands, population size, prob. dimensions)
    try to mimick migration with a network topology describe by the adjacency matrix
    isl_id[islands, population]
    best_idx[best ID]"""



    _ = [isl.wait() for isl in islands]

    for key, value in migration_config['selection_policy'].items():
        if value == True:
            selection_policy = key

    if np.mod(gen, migration_config['probability']) == 0 and migration_config['migration_on'] == True:
        """starting the migration as a function of the migration_scheme and selection_policy"""
        print("________________Migration________________")

        if selection_policy == 'isl_best_policy':
            migration = isl_best_policy(islands = islands)
            mig_adn = migration[0]

        if selection_policy == 'isl_worst_policy':
            migration = isl_worst_policy(islands = islands)
            mig_adn = migration[0]

        if selection_policy == 'isl_rand_policy':
            migration = isl_random_policy(islands = islands)
            mig_adn = migration[0]

        if selection_policy == 'archi_best_policy':
            migration = archi_best_policy(islands = islands)
            mig_adn = migration[0]

        islands = remove_nativ_pop(islands, adjacency_matrix, udas, prob)

        islands = sending_migr_pop(islands, adjacency_matrix, mig_adn, udas)

        print("___________________End___________________")


    return islands


def isl_best_policy(islands):
    """compute the best policy, the best objective function is chosen for every islands"""
    best_ID = np.array([isl.get_population().best_idx() for isl in islands])
    isl_obj = np.array([isl.get_population().get_f() for isl in islands])
    isl_adn = np.array([isl.get_population().get_x() for isl in islands])
    migration_obj = []
    migration_adn = []
    for i in np.arange(0, len(islands)):
        obj = isl_obj[i]
        adn = isl_adn[i]
        migration_obj.append(obj[best_ID[i]])
        migration_adn.append(adn[best_ID[i], :])

    migration_obj = np.squeeze(np.array([migration_obj]))
    migration_adn = np.squeeze(np.array([migration_adn]))

    return migration_adn, migration_obj

def isl_worst_policy(islands):
    """compute the worst policy, the worst objective function is chosen for every islands
    migration_obj[#islands]
    migration_adn[#islands, problem dimensions]"""
    worst_ID = np.array([isl.get_population().worst_idx() for isl in islands])
    isl_obj = np.array([isl.get_population().get_f() for isl in islands])
    isl_adn = np.array([isl.get_population().get_x() for isl in islands])
    migration_obj = []
    migration_adn = []
    for i in np.arange(0, len(islands)):
        migration_obj.append(isl_obj[i, worst_ID[i], 0])
        migration_adn.append(isl_adn[i, worst_ID[i], :])

    migration_obj = np.squeeze(np.array([migration_obj]))
    migration_adn = np.squeeze(np.array([migration_adn]))

    return migration_adn, migration_obj

def isl_random_policy(islands):
    """compute the random policy for each island"""
    isl_obj = np.array([isl.get_population().get_f() for isl in islands])
    isl_adn = np.array([isl.get_population().get_x() for isl in islands])
    migration_obj = []
    migration_adn = []
    for i in np.arange(0, len(islands)):
        rand = np.random.randint(0, len(isl_adn))
        migration_obj.append(isl_obj[i, rand])
        migration_adn.append(isl_adn[i, rand, :])

    migration_obj = np.squeeze(np.array([migration_obj]))
    migration_adn = np.squeeze(np.array([migration_adn]))

    return migration_adn, migration_obj

def archi_best_policy(islands):
    isl_obj = np.array([isl.get_population().get_f() for isl in islands])
    best_idx = np.array([isl.get_population().best_idx() for isl in islands])
    isl_best_obj = np.array([isl.get_population().champion_f for isl in islands])
    the_best_obj = np.amin(np.array([isl.get_population().champion_f for isl in islands]))
    id_best_isl = np.squeeze(np.where(isl_best_obj == the_best_obj))

    best_island = islands[id_best_isl[0]].get_population().get_x()
    best_indiv = best_idx[id_best_isl[0]]
    migration_obj = isl_obj[id_best_isl,best_indiv]
    migration_adn = best_island[best_indiv, :]

    return migration_adn, migration_obj

def draw_archi(adj_matrix, islands):
    plt.figure('Archipelago topology')
    plt.title('Archiepelago topology')
    rows, cols = np.where(adj_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    G = nx.DiGraph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G)
    labels = {}

    for key, value in pos.items():
        pos[key] = [np.cos((float(key) / len(islands)) * 2 * np.pi), np.sin((float(key) / len(islands)) * 2 * np.pi)]
        labels[key] = '{}'.format(key)

    nx.draw(G, pos=pos, node_size=700, node_alpha=0.1, node_color=(0.56, 0.62, 0.66),
            edge_color='black', graph_layout='shell', with_labels=True)
    plt.axis('equal')
    plt.show(block=False)
    return


def sending_migr_pop(islands, adjacency_matrix, mig_adn, udas):
    """Sending the individuals with respect to the adjacency  matrix"""
    for i in np.arange(0, len(islands)):
        for j in np.arange(0, len(islands)):
            if adjacency_matrix[i, j] == 1:
                pop = islands[j].get_population()
                pop.push_back(mig_adn[i, :])
                islands[j] = pg.island(algo=udas[j], pop=pop, udi=pg.thread_island())

    return islands


def remove_nativ_pop(islands, adjacency_matrix, udas, prob):
    """removing the native population with the help of selection policies"""
    nbr_removal = np.sum(adjacency_matrix, axis=0)
    pops = [isl.get_population() for isl in islands]
    for i in np.arange(0, len(islands)):
        obj = pops[i].get_f()
        adn = pops[i].get_x()
        id_sorted = np.argsort(obj, axis=0)
        sorted_obj = obj[id_sorted]
        sorted_adn = adn[id_sorted]

        # erase the population in the island i
        islands[i].set_population(pg.population(prob))
        pop = islands[i].get_population()
        for j in np.arange(0, int(len(pops[i])-nbr_removal[i])):
            #np.squeeze(sorted_adn[j, :]).tolist()
            pop.push_back(np.squeeze(sorted_adn[j,:]).tolist())#x=np.squeeze(sorted_adn[j, :]).tolist(),f=[np.squeeze(sorted_obj[j,:]).tolist()])

        islands[i] = pg.island(algo=udas[i], pop=pop, udi=pg.thread_island())

    return islands

def build_adjacency(islands, migration_config):
    """Construct the adjacency matrix of the island network"""

    for key, value in migration_config['scheme'].items():
        if value == True:
            scheme = key
            print(key)

    if scheme == 'ring':
        col = np.append(np.zeros(len(islands)-1), np.array([1])).reshape(len(islands), 1)
        adj_matrix = np.append(col, np.roll(col, 1), axis=1)
        for i in np.arange(2, len(islands)):
            adj_matrix = np.append(adj_matrix, np.roll(col, i), axis=1)

    if scheme == 'matrix':
        adj_matrix = np.array(migration_config['adjacency_matrix'])

    if scheme == 'rand':
        adj_matrix = np.zeros((len(islands),len(islands)))
        for i in np.arange(0, len(islands)):
            for j in np.arange(0, len(islands)):
                if i != j:
                    adj_matrix[i, j] = np.random.randint(0, 2)

    if scheme == 'full':
        col = np.append(np.array([0]), np.ones(len(islands) - 1)).reshape(len(islands), 1)
        adj_matrix = np.append(col, np.roll(col, 1), axis=1)
        for i in np.arange(2, len(islands)):
            adj_matrix = np.append(adj_matrix, np.roll(col, i), axis=1)

    if scheme == 'low_density':
        col1 = [np.append([0, 1], np.zeros(int(np.round(len(islands)/2)-1)), axis=0)]
        col1 = [np.append(col1, np.array([1]))]
        col1 = [np.append(col1, np.zeros(int(np.round(len(islands)/2)-2)))]
        col2 = [np.append([0, 0, 1], np.zeros(2*int(np.round(len(islands)/2))-3))]
        adj_matrix = np.append(col1, col2, axis=0)
        for i in np.arange(2,len(islands)):
            if i % 2 == 0:
                adj_matrix = np.append(adj_matrix, np.roll(col1, i), axis=0)
            else:
                adj_matrix = np.append(adj_matrix, np.roll(col2, i-1), axis=0)

    if scheme == 'single':
        col = [np.append(np.array([1]), np.zeros(len(islands)-1))]
        adj_matrix = np.append(np.roll(col, np.random.randint(0, len(islands))),
                               np.roll(col, np.random.randint(0, len(islands))),axis = 0)
        for i in np.arange(2,len(islands)):
            adj_matrix = np.append(adj_matrix, np.roll(col, np.random.randint(0,len(islands))),axis = 0)

    draw_archi(adj_matrix, islands)
    return adj_matrix