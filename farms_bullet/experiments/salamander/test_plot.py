import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# from farms_bullet.experiments.salamander.animat_data import *

from farms_bullet.experiments.salamander import animat_data
from farms_bullet.experiments.salamander.animat_options import SalamanderOptions, SalamanderOscillatorFrequenciesOptions
from farms_bullet.experiments.salamander.convention import bodyosc2index, legosc2index
import importlib

importlib.reload(animat_data)

n_joints = 11
n_dofs_leg = 4
n_leg = 4
dim_body = n_joints * 2
dim = 2 * n_joints + 2 * n_leg * n_dofs_leg
options = SalamanderOptions()
C = animat_data.SalamanderOscillatorConnectivityArray.set_options(options)
contact_array = animat_data.SalamanderContactsConnectivityArray.export_params()

array = np.array(C)

G = nx.DiGraph()
plt.figure()
pos = np.zeros([dim, 2])
scale_factor = 0.5
offset_leg = 1.5
for i in np.arange(dim):

    if i < dim_body:
        G.add_node(i, pos=(-scale_factor, -scale_factor * (i)))
        if i >= n_joints:
            G.add_node(i, pos=(scale_factor, -scale_factor * (i - n_joints)))

    if i < dim_body + n_dofs_leg and i >= dim_body:
        G.add_node(i, pos=(scale_factor * (-i + dim_body) - offset_leg, 0))
    if i < dim_body + 2 * n_dofs_leg and i >= dim_body + n_dofs_leg:
        G.add_node(i, pos=(scale_factor * (-i + dim_body + n_dofs_leg) - offset_leg, -scale_factor))
    if i < dim_body + 3 * n_dofs_leg and i >= dim_body + 2 * n_dofs_leg:
        G.add_node(i, pos=(scale_factor * (i - dim_body - 2 * n_dofs_leg) + offset_leg, 0))
    if i < dim_body + 4 * n_dofs_leg and i >= dim_body + 3 * n_dofs_leg:
        G.add_node(i, pos=(scale_factor * (i - dim_body - 3 * n_dofs_leg) + offset_leg, -scale_factor))
    if i < dim_body + 5 * n_dofs_leg and i >= dim_body + 4 * n_dofs_leg:
        G.add_node(i, pos=(scale_factor * (-i + dim_body + 4 * n_dofs_leg) - offset_leg, -2))
    if i < dim_body + 6 * n_dofs_leg and i >= dim_body + 5 * n_dofs_leg:
        G.add_node(i, pos=(scale_factor * (-i + dim_body + 5 * n_dofs_leg) - offset_leg, -2.5))
    if i < dim_body + 7 * n_dofs_leg and i >= dim_body + 6 * n_dofs_leg:
        G.add_node(i, pos=(scale_factor * (i - dim_body - 6 * n_dofs_leg) + offset_leg, -2))
    if i < dim_body + 8 * n_dofs_leg and i >= dim_body + 7 * n_dofs_leg:
        G.add_node(i, pos=(scale_factor * (i - dim_body - 7 * n_dofs_leg) + offset_leg, -2.5))

G.add_node(dim + 1, pos=(-5, -1), node_color='r')
G.add_node(dim + 2, pos=(5, -1))
G.add_node(dim + 3, pos=(-5, -2))
G.add_node(dim + 4, pos=(5, -2))

G.add_weighted_edges_from(array[:, 0:3], colors='k')

# G.add_weighted_edges_from(np.vstack((contact_array[:, 0], contact_array[:, 1] + 55, np.zeros(len(contact_array)))).T,
#                          colors='r')
# G.add_weighted_edges_from(contact_array[:, 0:3], colors='r')
for i in contact_array:
    G.add_weighted_edges_from([(i[0], i[1] + 55, 0)], edge_color='r')
graph_pose = nx.get_node_attributes(G, 'pos')
M = G.reverse()
# edges = G.edges()
colors = ['g'] * dim + ['r'] * 4
nx.draw(M, with_labels=True, node_color=colors, node_size=500, pos=graph_pose)
plt.axis('equal')
plt.show()
# plt.show(block=False)


print('------test indexing--------')
for leg_i in range(2):
    for side_i in range(2):
        for i in np.arange(0, 5):
            print(legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0), i)
for leg_i in range(2):
    for side_i in range(2):
        for i in np.arange(11, 16):
            print(legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1), i)

for leg_i in range(2):
    for side_i in range(2):
        for i in range(11):  # [0, 1, 7, 8, 9, 10]
            for side_leg in range(2):  # Muscle facing front/back
                for lateral in range(2):
                    walk_phase = (
                        np.pi
                        if i in [0, 1, 7, 8, 9, 10]
                        else 0
                    )
                    print(
                            walk_phase
                            + np.pi * (side_i + 1)
                            + lateral * np.pi
                            + side_leg * np.pi
                            + leg_i * np.pi
                    )
