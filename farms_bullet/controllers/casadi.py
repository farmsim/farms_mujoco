"""CasADi implementation"""

import numpy as np
import casadi as cas


def bodyjoint2index(joint_i):
    """body2index"""
    return joint_i


def legjoint2index(leg_i, side_i, joint_i, offset=11):
    """legjoint2index"""
    return offset + leg_i*3*2 + side_i*3 + joint_i


class CasADiNetwork:
    """Controller network"""

    def __init__(self, state, integrator):
        super(CasADiNetwork, self).__init__()
        self._state = state
        self._integrator = integrator

    @property
    def state(self):
        """State"""
        return self._state

    def integrate(self, parameters):
        """Control step"""
        self._state = np.array(
            self._integrator(
                x0=self._state,
                p=parameters
            )["xf"][:, 0]
        )


class IndependentCasADiOscillators(CasADiNetwork):
    """Independent oscillators"""

    def __init__(self, controllers, timestep):
        size = len(controllers)
        freqs = cas.MX.sym('freqs', size)
        ode = {
            "x": cas.MX.sym('phases', size),
            "p": freqs,
            "ode": freqs
        }
        super(IndependentCasADiOscillators, self).__init__(
            state=np.zeros(size),
            integrator=cas.integrator(
                'oscillator',
                'cvodes',
                ode,
                {
                    "t0": 0,
                    "tf": timestep,
                    "jit": True,
                    # "step0": 1e-3,
                    # "abstol": 1e-3,
                    # "reltol": 1e-3
                },
            )
        )

    @property
    def phases(self):
        """Oscillator phases"""
        return self._state

    def control_step(self, freqs):
        """Control step"""
        self.integrate(freqs)
        return self.phases


class SalamanderCasADiNetwork(CasADiNetwork):
    """Salamander network"""

    def __init__(self, phases, freqs, weights, phases_desired, integrator):
        self.freqs, self.weights, self.phases_desired = (
            freqs,
            weights,
            phases_desired
        )
        super(SalamanderCasADiNetwork, self).__init__(
            state=phases,
            integrator=integrator
        )

    @classmethod
    def from_gait(cls, gait, timestep, phases=None):
        """ Salamander network from gait"""
        return (
            cls.swimming(timestep, phases)
            if gait == "swimming"
            else cls.walking(timestep, phases)
        )

    @staticmethod
    def walking_parameters():
        """Walking parameters"""
        n_dim_body = 11
        n_dim_legs = 2*2*3
        n_dim = n_dim_body + n_dim_legs
        weights = np.zeros([n_dim, n_dim])
        phases_desired = np.zeros([n_dim, n_dim])
        # Body
        for i in range(10):
            weights[i, i+1] = 3e2
            weights[i+1, i] = 3e2
            phases_desired[i, i+1] = 0  # -2*np.pi/n_dim_body
            phases_desired[i+1, i] = 0  # 2*np.pi/n_dim_body
        # Legs
        for leg_i in range(2):
            for side_i in range(2):
                # 0 - 1
                weights[
                    legjoint2index(leg_i, side_i, 0),
                    legjoint2index(leg_i, side_i, 1)
                ] = 3e2
                weights[
                    legjoint2index(leg_i, side_i, 1),
                    legjoint2index(leg_i, side_i, 0)
                ] = 3e2
                phases_desired[
                    legjoint2index(leg_i, side_i, 0),
                    legjoint2index(leg_i, side_i, 1)
                ] = 0.5*np.pi
                phases_desired[
                    legjoint2index(leg_i, side_i, 1),
                    legjoint2index(leg_i, side_i, 0)
                ] = -0.5*np.pi
                # 1 - 2
                weights[
                    legjoint2index(leg_i, side_i, 1),
                    legjoint2index(leg_i, side_i, 2)
                ] = 3e2
                weights[
                    legjoint2index(leg_i, side_i, 2),
                    legjoint2index(leg_i, side_i, 1)
                ] = 3e2
                phases_desired[
                    legjoint2index(leg_i, side_i, 1),
                    legjoint2index(leg_i, side_i, 2)
                ] = 0
                phases_desired[
                    legjoint2index(leg_i, side_i, 2),
                    legjoint2index(leg_i, side_i, 1)
                ] = 0
        # # Opposite leg interaction
        # for leg_i in range(2):
        #     # 0 - 1
        #     weights[
        #         legjoint2index(leg_i, 0, 0),
        #         legjoint2index(leg_i, 1, 0)
        #     ] = 3e2
        #     weights[
        #         legjoint2index(leg_i, 1, 0),
        #         legjoint2index(leg_i, 0, 0)
        #     ] = 3e2
        #     phases_desired[
        #         legjoint2index(leg_i, 0, 0),
        #         legjoint2index(leg_i, 1, 0)
        #     ] = np.pi
        #     phases_desired[
        #         legjoint2index(leg_i, 1, 0),
        #         legjoint2index(leg_i, 0, 0)
        #     ] = -np.pi
        # # Following leg interaction
        # for side_i in range(2):
        #     # 0 - 1
        #     weights[
        #         legjoint2index(0, side_i, 0),
        #         legjoint2index(1, side_i, 0)
        #     ] = 3e2
        #     weights[
        #         legjoint2index(1, side_i, 0),
        #         legjoint2index(0, side_i, 0)
        #     ] = 3e2
        #     phases_desired[
        #         legjoint2index(0, side_i, 0),
        #         legjoint2index(1, side_i, 0)
        #     ] = np.pi
        #     phases_desired[
        #         legjoint2index(1, side_i, 0),
        #         legjoint2index(0, side_i, 0)
        #     ] = -np.pi
        # Body-legs interaction
        for side_i in range(2):
            # Forelimbs
            weights[
                bodyjoint2index(1),
                legjoint2index(0, side_i, 0)
            ] = 3e2
            weights[
                legjoint2index(0, side_i, 0),
                bodyjoint2index(1)
            ] = 3e2
            phases_desired[
                bodyjoint2index(1),
                legjoint2index(0, side_i, 0)
            ] = side_i*np.pi  # 0.5*np.pi
            phases_desired[
                legjoint2index(0, side_i, 0),
                bodyjoint2index(1)
            ] = -side_i*np.pi  # -0.5*np.pi
            # Hind limbs
            weights[
                bodyjoint2index(4),
                legjoint2index(1, side_i, 0)
            ] = 3e2
            weights[
                legjoint2index(1, side_i, 0),
                bodyjoint2index(4)
            ] = 3e2
            phases_desired[
                bodyjoint2index(4),
                legjoint2index(1, side_i, 0)
            ] = (side_i-1)*np.pi  # -0.5*np.pi
            phases_desired[
                legjoint2index(1, side_i, 0),
                bodyjoint2index(4)
            ] = (side_i-1)*np.pi  # 0.5*np.pi
        freqs = 2*np.pi*np.ones(n_dim_body)
        phases = 1e-3*3e-1*np.pi*(2*np.pi*np.random.ranf(n_dim)-1)
        return n_dim, phases, freqs, weights, phases_desired

    @classmethod
    def walking(cls, timestep, phases=None):
        """Default salamander network"""
        n_dim, _phases, freqs, weights, phases_desired = (
            cls.walking_parameters()
        )
        if phases is None:
            phases = _phases
        weights, phase_desired, integrator = cls.gen_cas_integrator(
            timestep,
            n_dim,
            weights,
            phases_desired
        )
        cls.walking_parameters()
        return cls(phases, freqs, weights, phase_desired, integrator)

    @classmethod
    def swimming(cls, timestep, phases=None):
        """Default salamander network"""
        n_dim_body = 11
        n_dim_legs = 2*2*3
        n_dim = n_dim_body + n_dim_legs
        weights = np.zeros([n_dim, n_dim])
        phases_desired = np.zeros([n_dim, n_dim])
        # Body
        for i in range(10):
            weights[i, i+1] = 3e2
            weights[i+1, i] = 3e2
            phases_desired[i, i+1] = 2*np.pi/n_dim_body
            phases_desired[i+1, i] = -2*np.pi/n_dim_body
        # Legs
        for leg_i in range(2):
            for side_i in range(2):
                # 0 - 1
                weights[
                    legjoint2index(leg_i, side_i, 0),
                    legjoint2index(leg_i, side_i, 1)
                ] = 3e2
                weights[
                    legjoint2index(leg_i, side_i, 1),
                    legjoint2index(leg_i, side_i, 0)
                ] = 3e2
                phases_desired[
                    legjoint2index(leg_i, side_i, 0),
                    legjoint2index(leg_i, side_i, 1)
                ] = 0
                phases_desired[
                    legjoint2index(leg_i, side_i, 1),
                    legjoint2index(leg_i, side_i, 0)
                ] = 0
                # 1 - 2
                weights[
                    legjoint2index(leg_i, side_i, 1),
                    legjoint2index(leg_i, side_i, 2)
                ] = 3e2
                weights[
                    legjoint2index(leg_i, side_i, 2),
                    legjoint2index(leg_i, side_i, 1)
                ] = 3e2
                phases_desired[
                    legjoint2index(leg_i, side_i, 1),
                    legjoint2index(leg_i, side_i, 2)
                ] = 0
                phases_desired[
                    legjoint2index(leg_i, side_i, 2),
                    legjoint2index(leg_i, side_i, 1)
                ] = 0
        # # Opposite leg interaction
        # for leg_i in range(2):
        #     # 0 - 1
        #     weights[
        #         legjoint2index(leg_i, 0, 0),
        #         legjoint2index(leg_i, 1, 0)
        #     ] = 3e2
        #     weights[
        #         legjoint2index(leg_i, 1, 0),
        #         legjoint2index(leg_i, 0, 0)
        #     ] = 3e2
        #     phases_desired[
        #         legjoint2index(leg_i, 0, 0),
        #         legjoint2index(leg_i, 1, 0)
        #     ] = np.pi
        #     phases_desired[
        #         legjoint2index(leg_i, 1, 0),
        #         legjoint2index(leg_i, 0, 0)
        #     ] = -np.pi
        # # Following leg interaction
        # for side_i in range(2):
        #     # 0 - 1
        #     weights[
        #         legjoint2index(0, side_i, 0),
        #         legjoint2index(1, side_i, 0)
        #     ] = 3e2
        #     weights[
        #         legjoint2index(1, side_i, 0),
        #         legjoint2index(0, side_i, 0)
        #     ] = 3e2
        #     phases_desired[
        #         legjoint2index(0, side_i, 0),
        #         legjoint2index(1, side_i, 0)
        #     ] = np.pi
        #     phases_desired[
        #         legjoint2index(1, side_i, 0),
        #         legjoint2index(0, side_i, 0)
        #     ] = -np.pi
        # Body-legs interaction
        for side_i in range(2):
            # Forelimbs
            weights[
                bodyjoint2index(1),
                legjoint2index(0, side_i, 0)
            ] = 3e2
            weights[
                legjoint2index(0, side_i, 0),
                bodyjoint2index(1)
            ] = 3e2
            phases_desired[
                bodyjoint2index(1),
                legjoint2index(0, side_i, 0)
            ] = 0  # 0.5*np.pi
            phases_desired[
                legjoint2index(0, side_i, 0),
                bodyjoint2index(1)
            ] = 0  # -0.5*np.pi
            # Hind limbs
            weights[
                bodyjoint2index(4),
                legjoint2index(1, side_i, 0)
            ] = 3e2
            weights[
                legjoint2index(1, side_i, 0),
                bodyjoint2index(4)
            ] = 3e2
            phases_desired[
                bodyjoint2index(4),
                legjoint2index(1, side_i, 0)
            ] = 0  # -0.5*np.pi
            phases_desired[
                legjoint2index(1, side_i, 0),
                bodyjoint2index(4)
            ] = 0  # 0.5*np.pi
        freqs = 2*np.pi*np.ones(n_dim_body)
        if phases is None:
            phases = 1e-3*3e-1*np.pi*(2*np.pi*np.random.ranf(n_dim)-1)
        weights, phase_desired, integrator = cls.gen_cas_integrator(
            timestep,
            n_dim,
            weights,
            phases_desired
        )
        return cls(phases, freqs, weights, phase_desired, integrator)

    @staticmethod
    def gen_cas_integrator(timestep, n_joints, weights, phases_desired):
        """Generate controller"""
        oscillator_names = np.array(
            [
                "body_{}".format(i)
                for i in range(11)
            ] + [
                "legs_{}_{}_{}".format(leg_i, side, joint_i)
                for leg_i in range(2)
                for side in ["L", "R"]
                for joint_i in range(3)
            ]
        )
        freqs_sym = np.array([
            cas.SX.sym(oscillator)
            for oscillator in oscillator_names
        ])
        phases_sym = np.array([
            [cas.SX.sym("phase_{}".format(oscillator))]
            for oscillator in oscillator_names
        ])
        coupling_weights_dense = np.array([
            [
                cas.SX.sym("w_{}_{}".format(oscillator_0, oscillator_1))
                if weights[i, j] != 0
                else 0  # cas.SX.sym("0")
                for j, oscillator_1 in enumerate(oscillator_names)
            ] for i, oscillator_0 in enumerate(oscillator_names)
        ])
        phases_desired_dense = np.array([
            [
                cas.SX.sym("theta_d_{}_{}".format(i, j))
                if weights[i, j] != 0
                else 0  # cas.SX.sym("0")
                for j in range(n_joints)
            ] for i in range(n_joints)
        ])
        print("phases:\n{}".format(phases_sym))
        phase_repeat = np.repeat(phases_sym, n_joints, axis=1)
        print("phases_repeat:\n{}".format(phase_repeat))
        phase_diff_sym = phase_repeat.T-phase_repeat
        print("phases_diff:\n{}".format(phase_diff_sym))
        ode = (
            freqs_sym + np.sum(
                coupling_weights_dense*np.sin(
                    phase_diff_sym + phases_desired_dense
                ),
                axis=1
            )
        )
        print("ODE:\n{}".format(ode))

        print("Phases:\n{}".format(phases_sym.T))
        print("Freqs:\n{}".format(freqs_sym))
        # print("Coupling weights:\n{}".format(coupling_weights))
        coupling_weights_sym = np.array([
            coupling_weights_dense[i, j]
            for i in range(n_joints)
            for j in range(n_joints)
            if isinstance(coupling_weights_dense[i, j], cas.SX)
        ])
        phases_desired_sym = np.array([
            phases_desired_dense[i, j]
            for i in range(n_joints)
            for j in range(n_joints)
            if isinstance(coupling_weights_dense[i, j], cas.SX)
        ])
        print("Coupling weights sym:\n{}".format(coupling_weights_sym))
        # Integrator
        dt = 1e-3
        # opts = {
        #     'tf': dt,
        #     # "nonlinear_solver_iteration": "functional",
        #     "grid": [0, dt],
        #     "ad_weight": 1.0,
        #     # "linear_multistep_method": "adams",
        #     # "fsens_all_at_once": True,
        #     # "max_multistep_order": 1,
        #     # "sensitivity_method": "staggered",
        #     # "linear_solver": "csparse",
        #     # "steps_per_checkpoint": 1,
        #     'jit': False,
        #     'jac_penalty': 0,
        #     "number_of_finite_elements": 1,
        #     "inputs_check": False,
        #     "enable_jacobian": True,
        #     "enable_reverse": True,
        #     "enable_fd":True,
        #     "print_time": False,
        #     "print_stats": False,
        #     # "reltol": 1e-3,
        #     # "abstol": 1e-3,
        #     'expand': True
        # }
        integrator = cas.integrator(
            'oscillator_network',
            'cvodes',
            # 'rk',
            {
                "x": phases_sym,
                "p": cas.vertcat(
                    freqs_sym,
                    coupling_weights_sym,
                    phases_desired_sym
                ),
                "ode": ode
            },
            {
                "t0": 0,
                "tf": timestep,
                "jit": False,
                # "step0": 1e-3,
                # "abstol": 1e-3,
                # "reltol": 1e-3
                # "enable_jacobian": False,
                # "number_of_finite_elements": 1
                # "print_stats": True,
                # "verbose": True,
                # "enable_fd": True,
                # "enable_jacobian": False,
                # "step0": 1e-4
            }
        )
        # integrator.print_options()
        # integrator.setOption("exact_jacobian", "false")
        weights = [
            weights[i, j]
            for i in range(n_joints)
            for j in range(n_joints)
            if isinstance(coupling_weights_dense[i, j], cas.SX)
        ]
        phases_desired = [
            phases_desired[i, j]
            for i in range(n_joints)
            for j in range(n_joints)
            if isinstance(coupling_weights_dense[i, j], cas.SX)
        ]
        return weights, phases_desired, integrator

    @property
    def phases(self):
        """Oscillator phases"""
        return self._state

    def control_step(self, freqs):
        """Control step"""
        self.integrate(np.concatenate(
            [
                freqs,
                self.weights,
                self.phases_desired
            ],
            axis=0
        ))
        return self.phases
