from control_options import SalamanderControlOptions
from farms_bullet.controllers.convention import bodyjoint2index, legjoint2index
import numpy as np

from control_options import SalamanderControlOptions
from farms_bullet.controllers.convention import bodyjoint2index, legjoint2index
import numpy as np

gait = 'walkin'
if gait == 'walking':
    n_body_joints = 11
    connectivity = []
    default_amplitude = 3e2

    # Amplitudes
    options = SalamanderControlOptions.walking()
    amplitudes = [
        options["body_stand_amplitude"] * np.sin(
            2 * np.pi * i / n_body_joints
            - options["body_stand_shift"]
        )
        for i in range(n_body_joints)
    ]

    # Body
    for i in range(n_body_joints - 1):
        # i - i
        connectivity.append([
            bodyjoint2index(joint_i=i, side=1),
            bodyjoint2index(joint_i=i, side=0),
            default_amplitude, np.pi
        ])
        connectivity.append([
            bodyjoint2index(joint_i=i, side=0),
            bodyjoint2index(joint_i=i, side=1),
            default_amplitude, np.pi
        ])
        # i - i+1
        phase_diff = (
            0
            if np.sign(amplitudes[i]) == np.sign(amplitudes[i + 1])
            else np.pi
        )
        for side in range(2):
            connectivity.append([
                bodyjoint2index(joint_i=i + 1, side=side),
                bodyjoint2index(joint_i=i, side=side),
                default_amplitude, phase_diff
            ])
            connectivity.append([
                bodyjoint2index(joint_i=i, side=side),
                bodyjoint2index(joint_i=i + 1, side=side),
                default_amplitude, phase_diff
            ])
    # i+1 - i+1 (final)
    connectivity.append([
        bodyjoint2index(joint_i=n_body_joints - 1, side=1),
        bodyjoint2index(joint_i=n_body_joints - 1, side=0),
        default_amplitude, np.pi
    ])
    connectivity.append([
        bodyjoint2index(joint_i=n_body_joints - 1, side=0),
        bodyjoint2index(joint_i=n_body_joints - 1, side=1),
        default_amplitude, np.pi
    ])

    # Legs (internal)
    for leg_i in range(2):
        for side_i in range(2):
            # 0 - 0
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                default_amplitude, np.pi
            ])
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                default_amplitude, np.pi
            ])
            # 0 - 1
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                default_amplitude, 0.5 * np.pi
            ])
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                default_amplitude, -0.5 * np.pi
            ])
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                default_amplitude, 0.5 * np.pi
            ])
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                default_amplitude, -0.5 * np.pi
            ])
            # 1 - 1
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                default_amplitude, np.pi
            ])
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                default_amplitude, np.pi
            ])
            # 1 - 2
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                default_amplitude, 0
            ])
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                default_amplitude, 0
            ])
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                default_amplitude, 0
            ])
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                default_amplitude, 0
            ])
            # 2 - 2
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                default_amplitude, np.pi
            ])
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                default_amplitude, np.pi
            ])

    # Opposite leg interaction
    # TODO

    # Following leg interaction
    # TODO

    # Body-legs interaction
    for side_i in range(2):
        # Forelimbs
        connectivity.append([
            legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=0),
            bodyjoint2index(joint_i=1, side=side_i),
            default_amplitude, np.pi
        ])
        connectivity.append([
            bodyjoint2index(joint_i=1, side=side_i),
            legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=0),
            default_amplitude, np.pi
        ])
        connectivity.append([
            legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=1),
            bodyjoint2index(joint_i=1, side=side_i),
            default_amplitude, 0
        ])
        connectivity.append([
            bodyjoint2index(joint_i=1, side=side_i),
            legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=1),
            default_amplitude, 0
        ])
        # Hind limbs
        connectivity.append([
            legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=0),
            bodyjoint2index(joint_i=4, side=side_i),
            default_amplitude, np.pi
        ])
        connectivity.append([
            bodyjoint2index(joint_i=4, side=side_i),
            legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=0),
            default_amplitude, np.pi
        ])
        connectivity.append([
            legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=1),
            bodyjoint2index(joint_i=4, side=side_i),
            default_amplitude, 0
        ])
        connectivity.append([
            bodyjoint2index(joint_i=4, side=side_i),
            legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=1),
            default_amplitude, 0
        ])

else:
    n_body_joints = 11
    connectivity = []
    default_amplitude = 3e2

    # Body
    for i in range(n_body_joints - 1):
        # i - i
        connectivity.append([
            bodyjoint2index(joint_i=i, side=1),
            bodyjoint2index(joint_i=i, side=0),
            default_amplitude, np.pi
        ])
        connectivity.append([
            bodyjoint2index(joint_i=i, side=0),
            bodyjoint2index(joint_i=i, side=1),
            default_amplitude, np.pi
        ])
        # i - i+1
        for side in range(2):
            connectivity.append([
                bodyjoint2index(joint_i=i + 1, side=side),
                bodyjoint2index(joint_i=i, side=side),
                default_amplitude, 2 * np.pi / n_body_joints
            ])
            connectivity.append([
                bodyjoint2index(joint_i=i, side=side),
                bodyjoint2index(joint_i=i + 1, side=side),
                default_amplitude, -2 * np.pi / n_body_joints
            ])
    # i+1 - i+1 (final)
    connectivity.append([
        bodyjoint2index(joint_i=n_body_joints - 1, side=1),
        bodyjoint2index(joint_i=n_body_joints - 1, side=0),
        default_amplitude, np.pi
    ])
    connectivity.append([
        bodyjoint2index(joint_i=n_body_joints - 1, side=0),
        bodyjoint2index(joint_i=n_body_joints - 1, side=1),
        default_amplitude, np.pi
    ])

    # Legs (internal)
    for leg_i in range(2):
        for side_i in range(2):
            # 0 - 0
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                default_amplitude, np.pi
            ])
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                default_amplitude, np.pi
            ])
            # 0 - 1
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                default_amplitude, 0.5 * np.pi
            ])
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                default_amplitude, -0.5 * np.pi
            ])
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                default_amplitude, 0.5 * np.pi
            ])
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                default_amplitude, -0.5 * np.pi
            ])
            # 1 - 1
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                default_amplitude, np.pi
            ])
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                default_amplitude, np.pi
            ])
            # 1 - 2
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                default_amplitude, 0
            ])
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                default_amplitude, 0
            ])
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                default_amplitude, 0
            ])
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                default_amplitude, 0
            ])
            # 2 - 2
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                default_amplitude, 0
            ])
            connectivity.append([
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
                legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
                default_amplitude, 0
            ])

    # Opposite leg interaction
    # TODO

    # Following leg interaction
    # TODO

    # Body-legs interaction
    for side_i in range(2):
        # Forelimbs
        connectivity.append([
            legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=0),
            bodyjoint2index(joint_i=1, side=side_i),
            default_amplitude, np.pi
        ])
        connectivity.append([
            bodyjoint2index(joint_i=1, side=side_i),
            legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=0),
            default_amplitude, np.pi
        ])
        connectivity.append([
            legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=1),
            bodyjoint2index(joint_i=1, side=side_i),
            default_amplitude, 0
        ])
        connectivity.append([
            bodyjoint2index(joint_i=1, side=side_i),
            legjoint2index(leg_i=0, side_i=side_i, joint_i=0, side=1),
            default_amplitude, 0
        ])
        # Hind limbs
        connectivity.append([
            legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=0),
            bodyjoint2index(joint_i=4, side=side_i),
            default_amplitude, np.pi
        ])
        connectivity.append([
            bodyjoint2index(joint_i=4, side=side_i),
            legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=0),
            default_amplitude, np.pi
        ])
        connectivity.append([
            legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=1),
            bodyjoint2index(joint_i=4, side=side_i),
            default_amplitude, 0
        ])
        connectivity.append([
            bodyjoint2index(joint_i=4, side=side_i),
            legjoint2index(leg_i=1, side_i=side_i, joint_i=0, side=1),
            default_amplitude, 0
        ])
connectivity = np.array(connectivity)
#np.savez('osc_connect', connectivity[:,0:2])
w_connect = np.load('osc_connect.npz')
my_connectivity = ()
#print(w_connect['arr_0'])
for i in w_connect['arr_0']:
    my_connectivity = my_connectivity + ((int(i[0]),int(i[1])),)

for j in np.arange(0,int(len(my_connectivity)/8)):
    print('{},{},{},{},{},{},{},{}'.format(my_connectivity[4*j],my_connectivity[4*j+1],my_connectivity[4*j+2],my_connectivity[4*j+3],my_connectivity[4*j+4],my_connectivity[4*j+5],my_connectivity[4*j+6],my_connectivity[4*j+7]))
