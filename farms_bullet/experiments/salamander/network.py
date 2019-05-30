"""Network"""

import numpy as np
from scipy import integrate
from .convention import bodyosc2index, legosc2index  # legjoint2index
from ...controllers.controller import ode_oscillators_sparse
from .animat_data import (
    SalamanderOscillatorNetworkState,
    # SalamanderNetworkParameters
    SalamanderData
)


class SalamanderNetworkODE:
    """Salamander network"""

    def __init__(self, state, animat_data, timestep):
        super(SalamanderNetworkODE, self).__init__()
        self.ode = ode_oscillators_sparse
        self.state = state
        self.animat_data = animat_data
        self._timestep = timestep
        self._n_oscillators = state.n_oscillators
        n_body = 11
        n_legs_dofs = 4
        self.groups = [None, None]
        self.groups[0] = [
            bodyosc2index(joint_i=i, side=0)
            for i in range(11)
        ] + [
            legosc2index(leg_i=leg_i, side_i=side_i, joint_i=joint_i, side=0)
            for leg_i in range(2)
            for side_i in range(2)
            for joint_i in range(n_legs_dofs)
        ]
        self.groups[1] = [
            bodyosc2index(joint_i=i, side=1)
            for i in range(n_body)
        ] + [
            legosc2index(leg_i=leg_i, side_i=side_i, joint_i=joint_i, side=1)
            for leg_i in range(2)
            for side_i in range(2)
            for joint_i in range(n_legs_dofs)
        ]

        # Adaptive timestep parameters
        self.solver = integrate.ode(f=self.ode)  # , jac=self.jac
        self.solver.set_integrator("dopri5")
        self.solver.set_f_params(self.animat_data)
        self._time = 0

    @classmethod
    def from_options(cls, options, n_iterations, timestep):
        """Salamander swimming network"""
        state = SalamanderOscillatorNetworkState.default_state(n_iterations)
        animat_data = SalamanderData.from_options(
            state,
            options,
            n_iterations
        )
        return cls(state, animat_data, timestep)

    def control_step(self):
        """Control step"""
        # Adaptive timestep (ODE)
        self.solver.set_initial_value(
            self.state.array[self.animat_data.iteration, 0, :],
            self._time
        )
        self._time += self._timestep
        self.state.array[self.animat_data.iteration+1, 0, :] = (
            self.solver.integrate(self._time)
        )
        self.animat_data.iteration += 1

        # # Adaptive timestep (ODEINT)
        # self.state.array[self.iteration+1, 0, :] = integrate.odeint(
        #     func=self.fun,
        #     Dfun=self.jac,
        #     y0=np.copy(self.state.array[self.iteration, 0, :]),
        #     t=np.linspace(0, self._timestep, 10),
        #     tfirst=True
        # )[-1]
        # self.iteration += 1

    @property
    def phases(self):
        """Oscillators phases"""
        return self.state.array[:, 0, :self._n_oscillators]

    @property
    def dphases(self):
        """Oscillators phases velocity"""
        return self.state.array[:, 1, :self._n_oscillators]

    @property
    def amplitudes(self):
        """Amplitudes"""
        return self.state.array[:, 0, self._n_oscillators:2*self._n_oscillators]

    @property
    def damplitudes(self):
        """Amplitudes velocity"""
        return self.state.array[:, 1, self._n_oscillators:2*self._n_oscillators]

    @property
    def offsets(self):
        """Offset"""
        return self.state.array[:, 0, 2*self._n_oscillators:]

    @property
    def doffsets(self):
        """Offset velocity"""
        return self.state.array[:, 1, 2*self._n_oscillators:]

    def get_outputs(self):
        """Outputs"""
        return self.amplitudes[self.animat_data.iteration]*(
            1 + np.cos(self.phases[self.animat_data.iteration])
        )

    def get_outputs_all(self):
        """Outputs"""
        return self.amplitudes*(
            1 + np.cos(self.phases)
        )

    def get_doutputs(self):
        """Outputs velocity"""
        return self.damplitudes[self.animat_data.iteration]*(
            1 + np.cos(self.phases[self.animat_data.iteration])
        ) - (
            self.amplitudes[self.animat_data.iteration]
            *np.sin(self.phases[self.animat_data.iteration])
            *self.dphases[self.animat_data.iteration]
        )

    def get_doutputs_all(self):
        """Outputs velocity"""
        return self.damplitudes*(
            1 + np.cos(self.phases)
        ) - self.amplitudes*np.sin(self.phases)*self.dphases

    def get_position_output(self):
        """Position output"""
        outputs = self.get_outputs()
        return (
            0.5*(outputs[self.groups[0]] - outputs[self.groups[1]])
            + self.offsets[self.animat_data.iteration]
        )

    def get_position_output_all(self):
        """Position output"""
        outputs = self.get_outputs_all()
        return (
            0.5*(outputs[:, self.groups[0]] - outputs[:, self.groups[1]])
            + self.offsets
        )

    def get_velocity_output(self):
        """Position output"""
        outputs = self.get_doutputs()
        return 0.5*(outputs[self.groups[0]] - outputs[self.groups[1]])

    def get_velocity_output_all(self):
        """Position output"""
        outputs = self.get_doutputs_all()
        return 0.5*(outputs[:, self.groups[0]] - outputs[:, self.groups[1]])

    def update(self, options):
        """Update drives"""
        self.animat_data.network.oscillators.update(options)
        self.animat_data.joints.update(options)
