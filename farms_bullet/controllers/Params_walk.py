from control_options import SalamanderControlOptions
#from ../animats.model_options import ModelOptions
import numpy as np
gait = "waking"

#walking parameters
#opt_mod = ModelOptions()

n_body = 11
n_dof_legs = 3
n_legs = 4
n_joints = n_body + n_legs * n_dof_legs
n_oscillators = 2 * (n_joints)
if gait == 'walking':
    freqs = 2 * np.pi * np.ones(n_oscillators)
    rates = 10 * np.ones(n_oscillators)
    options = SalamanderControlOptions.walking()
    # Amplitudes
    amplitudes = np.zeros(n_oscillators)
    for i in range(n_body):
        amplitudes[[i, i + n_body]] = np.abs(
            options["body_stand_amplitude"] * np.sin(
                2 * np.pi * i / n_body
                - options["body_stand_shift"]
                )
            )
        for leg_i in range(n_legs):
            for i in range(n_dof_legs):
                amplitudes[[
                2 * n_body + 2 * leg_i * n_dof_legs + i,
                2 * n_body + 2 * leg_i * n_dof_legs + i + n_dof_legs
                ]] = np.abs(
                    options["leg_{}_amplitude".format(i)]
                    )
else:

    freqs = 2 * np.pi * np.ones(n_oscillators)
    rates = 10 * np.ones(n_oscillators)
    amplitudes = np.zeros(n_oscillators)
    options = SalamanderControlOptions.swimming()
    body_amplitudes = np.linspace(
                options["body_amplitude_0"],
                options["body_amplitude_1"],
                n_body)
    for i in range(n_body):
        amplitudes[[i, i + n_body]] = body_amplitudes[i]
        for leg_i in range(n_legs):
            for i in range(n_dof_legs):
                        amplitudes[[
                        2 * n_body + 2 * leg_i * n_dof_legs + i,
                        2 * n_body + 2 * leg_i * n_dof_legs + i + n_dof_legs
                        ]] = (
                            options["leg_{}_amplitude".format(i)]
                            )

print(freqs, amplitudes)