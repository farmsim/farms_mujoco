"""Network"""

import numpy as np
from scipy import integrate
from .convention import bodyosc2index, legosc2index  # legjoint2index
from ...controllers.network import (
    ODE,
    ODESolver,
    OscillatorNetworkState,
    NetworkParameters,
    OscillatorArray,
    ConnectivityArray,
    JointsArray
)
from ...cy_controller import (
    rk4,
    # euler,
    ode_oscillators_sparse,
    ode_oscillators_sparse_gradient
)
from .animat_data import (
    SalamanderOscillatorNetworkState,
    SalamanderNetworkParameters
)


class SalamanderNetworkODE(ODESolver):
    """Salamander network"""

    def __init__(self, state, parameters, timestep):
        super(SalamanderNetworkODE, self).__init__(
            ode=ODE(
                solver=rk4,
                function=ode_oscillators_sparse,
                gradient=ode_oscillators_sparse_gradient
            ),
            state=state.array,
            timestep=timestep,
            parameters=parameters.to_ode_parameters()
        )
        self.state = state
        self.parameters = parameters
        self._n_oscillators = state.n_oscillators
        # self._n_joints = parameters.joints.shape()[1]
        n_body = 11
        n_legs_dofs = 4
        # n_legs = 4
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
        self.n_states = len(self.state.array[0, 0, :])
        self.dstate = np.zeros([self.n_states], dtype=np.float64)
        self._jac = np.zeros([self.n_states, self.n_states], dtype=np.float64)
        self.solver = integrate.ode(f=self.fun)  # , jac=self.jac
        self.solver.set_integrator("dopri5")
        self._time = 0
        self._parameters = self.parameters.to_ode_parameters().function

    @classmethod
    def from_options(cls, options, n_iterations, timestep):
        """Salamander swimming network"""
        state = SalamanderOscillatorNetworkState.default_state(n_iterations)
        parameters = SalamanderNetworkParameters.from_options(options)
        return cls(state, parameters, timestep)

    def fun(self, _time, state):
        """ODE function"""
        self.ode.function(
            self.dstate,
            state,
            *self._parameters
        )
        return self.dstate

    def jac(self, _time, state):
        """ODE function"""
        # self._jac = np.zeros([self.n_states, self.n_states], dtype=np.float64)
        self.ode.gradient(
            self._jac,
            state,
            *self._parameters
        )
        # np.set_printoptions(precision=3, linewidth=np.inf, threshold=np.inf)
        # print(self._jac)
        # raise Exception
        return self._jac

    @classmethod
    def from_gait(cls, gait, n_iterations, timestep):
        """ Salamander network from gait"""
        return (
            cls.swimming(n_iterations, timestep)
            if gait == "swimming"
            else cls.walking(n_iterations, timestep)
        )

    # def update_gait(self, gait):
    #     """Update from gait"""
    #     self.parameters.update_gait(gait)
    #     self._parameters = self.parameters.to_ode_parameters().function

    # @classmethod
    # def walking(cls, n_iterations, timestep):
    #     """Salamander swimming network"""
    #     state = SalamanderOscillatorNetworkState.default_state(n_iterations)
    #     parameters = SalamanderNetworkParameters.for_walking()
    #     return cls(state, parameters, timestep)

    # @classmethod
    # def swimming(cls, n_iterations, timestep):
    #     """Salamander swimming network"""
    #     state = SalamanderOscillatorNetworkState.default_state(n_iterations)
    #     parameters = SalamanderNetworkParameters.for_swimming()
    #     return cls(state, parameters, timestep)

    def control_step(self):
        """Control step"""
        # # Fixed timestep
        # self.step()

        # Adaptive timestep (ODE)
        self.solver.set_initial_value(
            self.state.array[self._iteration, 0, :],
            self._time
        )
        self._time += self._timestep
        self.state.array[self._iteration+1, 0, :] = (
            self.solver.integrate(self._time)
        )
        self.state.array[self._iteration+1, 1, :] = (
            self.state.array[self._iteration+1, 0, :]
            - self.state.array[self._iteration, 0, :]
        )/self._timestep
        self._iteration += 1

        # # Adaptive timestep (ODEINT)
        # self.state.array[self._iteration+1, 0, :] = integrate.odeint(
        #     func=self.fun,
        #     Dfun=self.jac,
        #     y0=np.copy(self.state.array[self._iteration, 0, :]),
        #     t=np.linspace(0, self._timestep, 10),
        #     tfirst=True
        # )[-1]
        # self._iteration += 1

    @property
    def phases(self):
        """Oscillators phases"""
        return self._state[:, 0, :self._n_oscillators]

    @property
    def dphases(self):
        """Oscillators phases velocity"""
        return self._state[:, 1, :self._n_oscillators]

    @property
    def amplitudes(self):
        """Amplitudes"""
        return self._state[:, 0, self._n_oscillators:2*self._n_oscillators]

    @property
    def damplitudes(self):
        """Amplitudes velocity"""
        return self._state[:, 1, self._n_oscillators:2*self._n_oscillators]

    @property
    def offsets(self):
        """Offset"""
        return self._state[:, 0, 2*self._n_oscillators:]

    @property
    def doffsets(self):
        """Offset velocity"""
        return self._state[:, 1, 2*self._n_oscillators:]

    def get_outputs(self):
        """Outputs"""
        return self.amplitudes[self.iteration]*(
            1 + np.cos(self.phases[self.iteration])
        )

    def get_outputs_all(self):
        """Outputs"""
        return self.amplitudes*(
            1 + np.cos(self.phases)
        )

    def get_doutputs(self):
        """Outputs velocity"""
        return self.damplitudes[self.iteration]*(
            1 + np.cos(self.phases[self.iteration])
        ) - (
            self.amplitudes[self.iteration]
            *np.sin(self.phases[self.iteration])
            *self.dphases[self.iteration]
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
            + self.offsets[self.iteration]
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
        self.parameters.oscillators.update(options)
        self.parameters.joints.update(options)
