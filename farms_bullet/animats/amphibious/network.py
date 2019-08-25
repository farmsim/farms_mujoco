"""Network"""

import numpy as np
from scipy import integrate
from .convention import bodyosc2index, legosc2index  # legjoint2index
from ...controllers.controller import ode_oscillators_sparse


class AmphibiousNetworkODE:
    """Amphibious network"""

    def __init__(self, animat_options, animat_data, timestep):
        super(AmphibiousNetworkODE, self).__init__()
        self.ode = ode_oscillators_sparse
        self.animat_options = animat_options
        self.animat_data = animat_data
        self._timestep = timestep
        self._n_oscillators = animat_data.state.n_oscillators
        n_body = self.animat_options.morphology.n_joints_body
        n_legs_dofs = self.animat_options.morphology.n_dof_legs
        self.groups = [None, None]
        self.groups = [
            [
                bodyosc2index(
                    joint_i=i,
                    side=side,
                    n_body_joints=animat_options.morphology.n_joints_body
                )
                for i in range(n_body)
            ] + [
                legosc2index(
                    leg_i=leg_i,
                    side_i=side_i,
                    joint_i=joint_i,
                    side=side,
                    n_legs=animat_options.morphology.n_legs,
                    n_body_joints=animat_options.morphology.n_joints_body,
                    n_legs_dof=animat_options.morphology.n_dof_legs

                )
                for leg_i in range(self.animat_options.morphology.n_legs//2)
                for side_i in range(2)
                for joint_i in range(n_legs_dofs)
            ]
            for side in range(2)
        ]

        # Adaptive timestep parameters
        self.solver = integrate.ode(f=self.ode)  # , jac=self.jac
        self.solver.set_integrator("dopri5")
        self.solver.set_f_params(self.animat_data)
        self._time = 0

    def control_step(self):
        """Control step"""
        # Adaptive timestep (ODE)
        self.solver.set_initial_value(
            self.animat_data.state.array[self.animat_data.iteration, 0, :],
            self._time
        )
        self._time += self._timestep
        self.animat_data.state.array[self.animat_data.iteration+1, 0, :] = (
            self.solver.integrate(self._time)
        )
        self.animat_data.iteration += 1

        # # Adaptive timestep (ODEINT)
        # self.animat_data.state.array[self.iteration+1, 0, :] = integrate.odeint(
        #     func=self.fun,
        #     Dfun=self.jac,
        #     y0=np.copy(self.animat_data.state.array[self.iteration, 0, :]),
        #     t=np.linspace(0, self._timestep, 10),
        #     tfirst=True
        # )[-1]
        # self.iteration += 1

    @property
    def phases(self):
        """Oscillators phases"""
        return self.animat_data.state.array[:, 0, :self._n_oscillators]

    @property
    def dphases(self):
        """Oscillators phases velocity"""
        return self.animat_data.state.array[:, 1, :self._n_oscillators]

    @property
    def amplitudes(self):
        """Amplitudes"""
        return self.animat_data.state.array[:, 0, self._n_oscillators:2*self._n_oscillators]

    @property
    def damplitudes(self):
        """Amplitudes velocity"""
        return self.animat_data.state.array[:, 1, self._n_oscillators:2*self._n_oscillators]

    @property
    def offsets(self):
        """Offset"""
        return self.animat_data.state.array[:, 0, 2*self._n_oscillators:]

    @property
    def doffsets(self):
        """Offset velocity"""
        return self.animat_data.state.array[:, 1, 2*self._n_oscillators:]

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
        return (
            0.5*(outputs[self.groups[0]] - outputs[self.groups[1]])
            + self.doffsets[self.animat_data.iteration]
        )

    def get_velocity_output_all(self):
        """Position output"""
        outputs = self.get_doutputs_all()
        return 0.5*(outputs[:, self.groups[0]] - outputs[:, self.groups[1]])

    def get_torque_output(self):
        """Torque output"""
        iteration = self.animat_data.iteration-1
        proprioception = self.animat_data.sensors.proprioception
        positions = np.array(proprioception.positions(iteration))
        velocities = np.array(proprioception.velocities(iteration))
        predicted_positions = (positions+3*self._timestep*velocities)
        cmd_positions = self.get_position_output()
        cmd_velocities = self.get_velocity_output()
        positions_rest = np.array(self.offsets[self.animat_data.iteration])
        cmd_kp = 1e1  # Nm/rad
        cmd_kd = 1e-2  # Nm*s/rad
        spring = 1e0  # Nm/rad
        damping = 1e-2  # Nm*s/rad
        max_torque = 1  # Nm
        torques = np.clip(
            (
                + cmd_kp*(cmd_positions-predicted_positions)
                + cmd_kd*(cmd_velocities-velocities)
                + spring*(positions_rest-predicted_positions)
                - damping*velocities
            ),
            -max_torque,
            +max_torque
        )
        return torques

    def update(self, options):
        """Update drives"""
        self.animat_data.network.oscillators.update(options)
        self.animat_data.joints.update(options)
