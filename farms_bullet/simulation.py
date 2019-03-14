"""Salamander simulation with pybullet"""

import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

import casadi as cas

import pybullet_data
import pybullet


def parse_args():
    """ Parse arguments """
    parser = argparse.ArgumentParser(description='Salamander simulation')
    parser.add_argument(
        '-f', '--free_camera',
        action='store_true',
        dest='free_camera',
        default=False,
        help='Allow for free camera (User controlled)'
    )
    parser.add_argument(
        '-r', '--rotating_camera',
        action='store_true',
        dest='rotating_camera',
        default=False,
        help='Enable rotating camera'
    )
    parser.add_argument(
        '-t', '--top_camera',
        action='store_true',
        dest='top_camera',
        default=False,
        help='Enable top view camera'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        dest='fast',
        default=False,
        help='Remove real-time limiter'
    )
    parser.add_argument(
        '--record',
        action='store_true',
        dest='record',
        default=False,
        help='Record video'
    )
    return parser.parse_args()


def bodyjoint2index(joint_i):
    """body2index"""
    return joint_i


def legjoint2index(leg_i, side_i, joint_i, offset=11):
    """legjoint2index"""
    return offset + leg_i*3*2 + side_i*3 + joint_i


class Network:
    """Controller network"""

    def __init__(self, state, integrator):
        super(Network, self).__init__()
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


class IndependentOscillators(Network):
    """Independent oscillators"""

    def __init__(self, controllers, timestep):
        size = len(controllers)
        freqs = cas.MX.sym('freqs', size)
        ode = {
            "x": cas.MX.sym('phases', size),
            "p": freqs,
            "ode": freqs
        }
        super(IndependentOscillators, self).__init__(
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


class SalamanderNetwork(Network):
    """Salamander network"""

    def __init__(self, phases, freqs, weights, phases_desired, integrator):
        self.freqs, self.weights, self.phases_desired = (
            freqs,
            weights,
            phases_desired
        )
        super(SalamanderNetwork, self).__init__(
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

    @classmethod
    def walking(cls, timestep, phases=None):
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
        if phases is None:
            phases = 1e-3*3e-1*np.pi*(2*np.pi*np.random.ranf(n_dim)-1)
        weights, phase_desired, integrator = cls.gen_integrator(
            timestep,
            n_dim,
            weights,
            phases_desired
        )
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
        weights, phase_desired, integrator = cls.gen_integrator(
            timestep,
            n_dim,
            weights,
            phases_desired
        )
        return cls(phases, freqs, weights, phase_desired, integrator)

    @staticmethod
    def gen_integrator(timestep, n_joints, weights, phases_desired):
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
        integrator = cas.integrator(
            'oscillator_network',
            'cvodes',
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
                "jit": True,
                # "step0": 1e-3,
                # "abstol": 1e-3,
                # "reltol": 1e-3
            }
        )
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


class SineControl:
    """SineControl"""

    def __init__(self, amplitude, frequency, offset):
        super(SineControl, self).__init__()
        self.amplitude = amplitude
        self._angular_frequency = 2*np.pi*frequency
        self.offset = offset

    @property
    def angular_frequency(self):
        """Angular frequency"""
        return self._angular_frequency

    @angular_frequency.setter
    def angular_frequency(self, value):
        self._angular_frequency = value

    def position(self, phase):
        """"Position"""
        return self.amplitude*np.sin(phase) + self.offset

    def velocity(self, phase):
        """Velocity"""
        return self._angular_frequency*self.amplitude*np.cos(phase)


class ControlPDF(dict):
    """ControlPDF"""

    def __init__(self, p=1, d=0, f=0):
        super(ControlPDF, self).__init__()
        self["p"] = p
        self["d"] = d
        self["f"] = f

    @property
    def p_term(self):
        """Proportfonal term"""
        return self["p"]

    @property
    def d_term(self):
        """Derivative term"""
        return self["d"]

    @property
    def f_term(self):
        """Max force term"""
        return self["f"]


class JointController:
    """JointController"""

    def __init__(self, joint, sine, pdf, **kwargs):
        super(JointController, self).__init__()
        self._joint = joint
        self._sine = sine
        self._pdf = pdf
        self._is_body = kwargs.pop("is_body", False)

    def cmds(self, phase):
        """Commands"""
        return {
            "pos": self._sine.position(phase),
            "vel": self._sine.velocity(phase)
        }

    def update(self, phase):
        """Update"""
        return {
            "joint": self._joint,
            "cmd": self.cmds(phase),
            "pdf": self._pdf
        }

    def angular_frequency(self):
        """Angular frequency"""
        return self._sine.angular_frequency

    def get_frequency(self):
        """Get frequency"""
        return self._sine.angular_frequency/(2*np.pi)

    def set_frequency(self, frequency):
        """Set frequency"""
        self._sine.angular_frequency = 2*np.pi*frequency

    def set_body_offset(self, body_offset):
        """Set body offset"""
        if self._is_body:
            self._sine.offset = body_offset


class ModelController:
    """ModelController"""

    def __init__(self, model, joints_controllers, timestep):
        super(ModelController, self).__init__()
        self.model = model
        self.controllers = joints_controllers
        # self.network = IndependentOscillators(
        #     self.controllers,
        #     timestep=timestep
        # )
        self.network = SalamanderNetwork.walking(timestep, phases=None)
        self._frequency = self.controllers[0].get_frequency()
        self._body_offset = 0

    def control(self, verbose=False):
        """Control"""
        phases = self.network.control_step([
            float(controller.angular_frequency())
            for controller in self.controllers
        ])
        if verbose:
            tic = time.time()
        controls = [
            controller.update(phases[i])
            for i, controller in enumerate(self.controllers)
        ]
        if verbose:
            toc = time.time()
            print("Time to copy phases: {} [s]".format(toc-tic))
        pybullet.setJointMotorControlArray(
            self.model,
            [ctrl["joint"] for ctrl in controls],
            pybullet.POSITION_CONTROL,
            targetPositions=[ctrl["cmd"]["pos"] for ctrl in controls],
            targetVelocities=[ctrl["cmd"]["vel"] for ctrl in controls],
            positionGains=[ctrl["pdf"]["p"] for ctrl in controls],
            velocityGains=[ctrl["pdf"]["d"] for ctrl in controls],
            forces=[ctrl["pdf"]["f"] for ctrl in controls]
        )

    def update_frequency(self, frequency):
        """Update frequency"""
        self._frequency = frequency
        for controller in self.controllers:
            controller.set_frequency(frequency)

    def update_body_offset(self, body_offset):
        """Update body offset"""
        self._body_offset = body_offset
        for controller in self.controllers:
            controller.set_body_offset(body_offset)


class SalamanderControlOptions(dict):
    """Model options"""

    def __init__(self, options):
        super(SalamanderControlOptions, self).__init__()
        self.update(options)

    @classmethod
    def from_gait(cls, gait, **kwargs):
        """Salamander control option from gait"""
        return (
            cls.walking(frequency=kwargs.pop("frequency", 1), **kwargs)
            if gait == "walking"
            else cls.swimming(frequency=kwargs.pop("frequency", 2), **kwargs)
            if gait == "swimming"
            else cls.standing()
        )

    @classmethod
    def standing(cls, **kwargs):
        """Standing options"""
        # Options
        options = {}

        # General
        options["n_body_joints"] = 11
        options["frequency"] = 0

        # Body
        options["body_amplitude_0"] = 0
        options["body_amplitude_1"] = 0
        options["body_stand_amplitude"] = 0
        options["body_stand_shift"] = 0

        # Legs
        options["leg_0_amplitude"] = 0
        options["leg_0_offset"] = 0

        options["leg_1_amplitude"] = 0
        options["leg_1_offset"] = np.pi/16

        options["leg_2_amplitude"] = 0
        options["leg_2_offset"] = np.pi/8

        # Additional walking options
        options["leg_turn"] = 0

        # Gains
        options["body_p"] = 1e-1
        options["body_d"] = 1e0
        options["body_f"] = 1e1
        options["legs_p"] = 1e-1
        options["legs_d"] = 1e0
        options["legs_f"] = 1e1

        # Additional options
        options.update(kwargs)
        return cls(options)

    @classmethod
    def walking(cls, **kwargs):
        """Walking options"""
        # Options
        options = {}

        # General
        options["n_body_joints"] = 11
        options["frequency"] = 1

        # Body
        options["body_amplitude_0"] = 0.0
        options["body_amplitude_1"] = 0.0
        options["body_stand_amplitude"] = 0.2
        options["body_stand_shift"] = np.pi/4

        # Legs
        options["leg_0_amplitude"] = 0.8
        options["leg_0_offset"] = 0

        options["leg_1_amplitude"] = np.pi/32
        options["leg_1_offset"] = np.pi/32

        options["leg_2_amplitude"] = np.pi/8
        options["leg_2_offset"] = np.pi/8

        # Additional walking options
        options["leg_turn"] = 0

        # Gains
        options["body_p"] = 1e-1
        options["body_d"] = 1e0
        options["body_f"] = 1e1
        options["legs_p"] = 1e-1
        options["legs_d"] = 1e0
        options["legs_f"] = 1e1

        # Additional options
        options.update(kwargs)
        return cls(options)

    @classmethod
    def swimming(cls, **kwargs):
        """Swimming options"""
        # Options
        options = {}

        # General
        n_body_joints = 11
        options["n_body_joints"] = n_body_joints
        options["frequency"] = 1

        # Body
        options["body_amplitude_0"] = 0.1
        options["body_amplitude_1"] = 0.5
        options["body_stand_amplitude"] = 0
        options["body_stand_shift"] = 0

        # Legs
        options["leg_0_amplitude"] = 0
        options["leg_0_offset"] = -2*np.pi/5

        options["leg_1_amplitude"] = 0
        options["leg_1_offset"] = 0

        options["leg_2_amplitude"] = 0
        options["leg_2_offset"] = 0

        # Additional walking options
        options["leg_turn"] = 0

        # Gains
        options["body_p"] = 1e-1
        options["body_d"] = 1e0
        options["body_f"] = 1e1
        options["legs_p"] = 1e-1
        options["legs_d"] = 1e0
        options["legs_f"] = 1e1

        # Additional options
        options.update(kwargs)
        return cls(options)

    def to_vector(self):
        """To vector"""
        return [
            self["frequency"],
            self["body_amplitude_0"],
            self["body_amplitude_1"],
            self["body_stand_amplitude"],
            self["body_stand_shift"],
            self["leg_0_amplitude"],
            self["leg_0_offset"],
            self["leg_1_amplitude"],
            self["leg_1_offset"],
            self["leg_2_amplitude"],
            self["leg_2_offset"],
            self["leg_turn"],
            self["body_p"],
            self["body_d"],
            self["body_f"],
            self["legs_p"],
            self["legs_d"],
            self["legs_f"]
        ]

    def from_vector(self, vector):
        """From vector"""
        (
            self["frequency"],
            self["body_amplitude_0"],
            self["body_amplitude_1"],
            self["body_stand_amplitude"],
            self["body_stand_shift"],
            self["leg_0_amplitude"],
            self["leg_0_offset"],
            self["leg_1_amplitude"],
            self["leg_1_offset"],
            self["leg_2_amplitude"],
            self["leg_2_offset"],
            self["leg_turn"],
            self["body_p"],
            self["body_d"],
            self["body_f"],
            self["legs_p"],
            self["legs_d"],
            self["legs_f"]
        ) = vector


class SalamanderController(ModelController):
    """ModelController"""

    @classmethod
    def from_gait(cls, model, joints, gait, timestep, **kwargs):
        """Salamander controller from gait"""
        return cls.from_options(
            model=model,
            joints=joints,
            options=SalamanderControlOptions.from_gait(gait, **kwargs),
            timestep=timestep
        )

    def update_gait(self, gait, joints, timestep):
        """Update gait"""
        controllers_body, controllers_legs = (
            SalamanderController.joints_controllers(
                joints=joints,
                options=SalamanderControlOptions.from_gait(
                    gait=gait,
                    frequency=self._frequency,
                    body_offset=self._body_offset
                )
            )
        )
        self.controllers = controllers_body + controllers_legs
        self.network = SalamanderNetwork.from_gait(
            gait,
            timestep,
            phases=self.network.phases
        )

    @classmethod
    def from_options(cls, model, joints, options, timestep):
        """Salamander controller from options"""
        joint_controllers_body, joint_controllers_legs = (
            cls.joints_controllers(joints, options)
        )
        return cls(
            model,
            joint_controllers_body + joint_controllers_legs,
            timestep=timestep
        )

    @staticmethod
    def joints_controllers(joints, options):
        """Controllers"""
        n_body_joints = options["n_body_joints"]
        frequency = options["frequency"]
        amplitudes = np.linspace(
            options["body_amplitude_0"],
            options["body_amplitude_1"],
            n_body_joints
        )
        joint_controllers_body = [
            JointController(
                joint=joints["joint_link_body_{}".format(joint_i+1)],
                sine=SineControl(
                    amplitude=amplitudes[joint_i] + (
                        options["body_stand_amplitude"]*np.sin(
                            2*np.pi*joint_i/n_body_joints
                            - options["body_stand_shift"]
                        )
                    ),
                    frequency=frequency,
                    offset=0
                ),
                pdf=(
                    ControlPDF(
                        p=options["body_p"],
                        d=options["body_d"],
                        f=options["body_f"]
                    )
                ),
                is_body=True
            )
            for joint_i in range(n_body_joints)
        ]
        joint_controllers_legs = [
            JointController(
                joint=joints["joint_link_leg_{}_{}_{}".format(
                    leg_i,
                    side,
                    joint_i
                )],
                sine=SineControl(
                    amplitude=options["leg_{}_amplitude".format(joint_i)],
                    frequency=frequency,
                    # phase=(
                    #     - np.pi*np.abs(leg_i-side_i)
                    #     - options["leg_{}_phase".format(joint_i)]
                    #     + options["leg_turn"]*float(  # Turning
                    #         (0.5)*np.pi*np.sign(np.abs(leg_i-side_i) - 0.5)
                    #         if joint_i == 2
                    #         else 0
                    #     )
                    # ),
                    offset=options["leg_{}_offset".format(joint_i)]
                ),
                pdf=ControlPDF(
                    p=options["legs_p"],
                    d=options["legs_d"],
                    f=options["legs_f"]
                )
            )
            for leg_i in range(2)
            for side_i, side in enumerate(["L", "R"])
            for joint_i in range(3)
        ]
        return joint_controllers_body, joint_controllers_legs


def init_engine():
    """Initialise engine"""
    print(pybullet.getAPIVersion())
    pybullet.connect(
        pybullet.GUI,
        # options="--enable_experimental_opencl"
        # options="--opengl2"  #  --minGraphicsUpdateTimeMs=32000
    )
    pybullet_path = pybullet_data.getDataPath()
    print("Adding pybullet data path {}".format(pybullet_path))
    pybullet.setAdditionalSearchPath(pybullet_path)


def viscous_swimming(model, links):
    """Viscous swimming"""
    # Swimming
    forces_torques = np.zeros([2, 10, 3])
    for link_i in range(1, 11):
        link_state = pybullet.getLinkState(
            model,
            links["link_body_{}".format(link_i)],
            computeLinkVelocity=1,
            computeForwardKinematics=1
        )
        link_orientation_inv = np.linalg.inv(np.array(
            pybullet.getMatrixFromQuaternion(link_state[5])
        ).reshape([3, 3]))
        link_velocity = np.dot(link_orientation_inv, link_state[6])
        link_angular_velocity = np.dot(link_orientation_inv, link_state[7])
        forces_torques[0, link_i-1, :] = (
            np.array([-1e-1, -1e0, -1e0])*link_velocity
        )
        pybullet.applyExternalForce(
            model,
            links["link_body_{}".format(link_i)],
            forceObj=forces_torques[0, link_i-1, :],
            posObj=[0, 0, 0],
            flags=pybullet.LINK_FRAME
        )
        forces_torques[1, link_i-1, :] = (
            np.array([-1e-2, -1e-2, -1e-2])*link_angular_velocity
        )
        pybullet.applyExternalTorque(
            model,
            links["link_body_{}".format(link_i+1)],
            torqueObj=forces_torques[1, link_i-1, :],
            flags=pybullet.LINK_FRAME
        )
    return forces_torques


def test_debug_info():
    """Test debug info"""
    line = pybullet.addUserDebugLine(
        lineFromXYZ=[0, 0, -0.09],
        lineToXYZ=[-3, 0, -0.09],
        lineColorRGB=[0.1, 0.5, 0.9],
        lineWidth=10,
        lifeTime=0
    )
    text = pybullet.addUserDebugText(
        text="BIOROB",
        textPosition=[-3, 0.1, -0.09],
        textColorRGB=[0, 0, 0],
        textSize=1,
        lifeTime=0,
        textOrientation=[0, 0, 0, 1],
        # parentObjectUniqueId
        # parentLinkIndex
        # replaceItemUniqueId
    )
    return line, text


def real_time_handing(timestep, tic_rt, toc_rt, rtl=1.0, **kwargs):
    """Real-time handling"""
    sleep_rtl = timestep/rtl - (toc_rt - tic_rt)
    rtf = timestep / (toc_rt - tic_rt)
    tic = time.time()
    sleep_rtl = np.clip(sleep_rtl, a_min=0, a_max=1)
    if sleep_rtl > 0:
        while time.time() - tic < sleep_rtl:
            time.sleep(0.1*sleep_rtl)
    if rtf < 0.5:
        print("Significantly slower than real-time: {} %".format(100*rtf))
        time_plugin = kwargs.pop("time_plugin", False)
        time_control = kwargs.pop("time_control", False)
        time_sim = kwargs.pop("time_sim", False)
        if time_plugin:
            print("  Time in py_plugins: {} [ms]".format(time_plugin))
        if time_control:
            print("    Time in control: {} [ms]".format(time_control))
        if time_sim:
            print("  Time in simulation: {} [ms]".format(time_sim))


def create_scene(plane):
    """Create scene"""

    # pybullet.createCollisionShape(pybullet.GEOM_PLANE)
    pybullet.createMultiBody(0,0)

    sphereRadius = 0.01
    colSphereId = pybullet.createCollisionShape(
        pybullet.GEOM_SPHERE,
        radius=sphereRadius
    )
    colCylinderId = pybullet.createCollisionShape(
        pybullet.GEOM_CYLINDER,
        radius=sphereRadius,
        height=1
    )
    colBoxId = pybullet.createCollisionShape(
        pybullet.GEOM_BOX,
        halfExtents=[sphereRadius, sphereRadius, sphereRadius]
    )

    mass = 1
    visualShapeId = -1


    link_Masses=[1]
    linkCollisionShapeIndices=[colBoxId]
    linkVisualShapeIndices=[-1]
    linkPositions=[[0,0,0.11]]
    linkOrientations=[[0,0,0,1]]
    linkInertialFramePositions=[[0,0,0]]
    linkInertialFrameOrientations=[[0,0,0,1]]
    indices=[0]
    jointTypes=[pybullet.JOINT_REVOLUTE]
    axis=[[0,0,1]]

    j = 0
    k = 0
    for i in range (30):
        for j in range (10):
            basePosition = [
                -3 - i*10*sphereRadius,
                -0.5 + j*10*sphereRadius,
                sphereRadius/2
            ]
            baseOrientation = [0, 0, 0, 1]
            sphereUid = pybullet.createMultiBody(
                mass,
                colCylinderId,
                visualShapeId,
                basePosition,
                baseOrientation
            )
            cid = pybullet.createConstraint(
                sphereUid, -1,
                plane, -1,
                pybullet.JOINT_FIXED,
                [0, 0, 1],
                [0, 0, 0],
                basePosition
            )

            pybullet.changeDynamics(
                sphereUid, -1,
                spinningFriction=0.001,
                rollingFriction=0.001,
                linearDamping=0.0
            )


class Simulation:
    """Simulation"""

    def __init__(self, timestep, duration, clargs, gait="walking"):
        super(Simulation, self).__init__()
        # Initialise engine
        init_engine()
        rendering(0)

        # Parameters
        # gait = "standing"
        self.gait = gait
        self.frequency = 1
        # gait = "swimming"
        self.timestep = timestep
        self.times = np.arange(0, duration, self.timestep)

        # Initialise
        self.model, self.plane = self.init_simulation(gait=gait)
        self.init(clargs)
        rendering(1)

    def get_entities(self):
        """Get simulation entities"""
        return (
            self.model,
            self.model.links,
            self.model.joints,
            self.plane.identity
        )

    def init_simulation(self, gait="walking"):
        """Initialise simulation"""
        # Physics
        self.init_physics(gait)

        # Spawn models
        model = SalamanderModel.spawn(self.timestep, gait)
        plane = Model.from_urdf(
            "plane.urdf",
            basePosition=[0, 0, -0.1]
        )
        return model, plane

    def init_physics(self, gait="walking"):
        """Initialise physics"""
        pybullet.resetSimulation()
        pybullet.setGravity(0, 0, -1e-2 if gait == "swimming" else -9.81)
        pybullet.setTimeStep(self.timestep)
        pybullet.setRealTimeSimulation(0)
        pybullet.setPhysicsEngineParameter(
            fixedTimeStep=self.timestep,
            numSolverIterations=50,
            erp=0,
            contactERP=0,
            frictionERP=0
        )
        print("Physics parameters:\n{}".format(
            pybullet.getPhysicsEngineParameters()
        ))

    def init(self, clargs):
        """Initialise simulation"""

        # Simulation entities
        self.salamander, self.links, self.joints, self.plane = (
            self.get_entities()
        )

        # Remove leg collisions
        self.salamander.leg_collisions(self.plane, activate=False)

        # Model information
        self.salamander.print_dynamics_info()

        # Create scene
        add_obstacles = False
        if add_obstacles:
            create_scene(plane)

        # Camera
        self.camera = UserCamera(
            target_identity=self.salamander.identity,
            yaw=0,
            yaw_speed=360/10 if clargs.rotating_camera else 0,
            pitch=-89 if clargs.top_camera else -45,
            distance=1,
            timestep=self.timestep
        )

        # Video recording
        if clargs.record:
            self.camera_record = CameraRecord(
                target_identity=self.salamander.identity,
                size=len(self.times)//25,
                fps=40,
                yaw=0,
                yaw_speed=360/10 if clargs.rotating_camera else 0,
                pitch=-89 if clargs.top_camera else -45,
                distance=1,
                timestep=self.timestep*25,
                motion_filter=1e-1
            )

        # User parameters
        self.user_params = UserParameters(self.gait, self.frequency)

        # Debug info
        test_debug_info()

        # Simulation time
        self.tot_plugin_time = 0
        self.tot_sim_time = 0
        self.tot_ctrl_time = 0
        self.tot_sensors_time = 0
        self.tot_log_time = 0
        self.tot_camera_time = 0
        self.tot_waitrt_time = 0
        self.forces_torques = np.zeros([len(self.times), 2, 10, 3])
        self.sim_step = 0

        # Final setup
        self.experiment_logger = ExperimentLogger(
            self.salamander,
            len(self.times)
        )
        self.init_state = pybullet.saveState()
        rendering(1)

    def run(self, clargs):
        """Run simulation"""
        # Run simulation
        self.tic = time.time()
        loop_time = 0
        while self.sim_step < len(self.times):
            if not(self.sim_step % 100):
                self.user_params.update()
                keys = pybullet.getKeyboardEvents()
                if ord("q") in keys:
                    break
            if not(self.sim_step % 10000) and self.sim_step > 0:
                pybullet.restoreState(self.init_state)
            if not self.user_params.play.value:
                time.sleep(0.5)
            else:
                tic_loop = time.time()
                self.loop(clargs)
                loop_time += time.time() - tic_loop
        print("Loop time: {} [s]".format(loop_time))
        self.toc = time.time()
        self.times_simulated = self.times[:self.sim_step]

    def loop(self, clargs):
        """Simulation loop"""
        self.tic_rt = time.time()
        self.sim_time = self.timestep*self.sim_step
        # Control
        if self.user_params.gait.changed:
            self.gait = self.user_params.gait.value
            self.model.controller.update_gait(
                self.gait,
                self.joints,
                self.timestep
            )
            pybullet.setGravity(
                0, 0, -1e-2 if self.gait == "swimming" else -9.81
            )
            self.user_params.gait.changed = False
        if self.user_params.frequency.changed:
            self.model.controller.update_frequency(
                self.user_params.frequency.value
            )
            self.user_params.frequency.changed = False
        if self.user_params.body_offset.changed:
            self.model.controller.update_body_offset(
                self.user_params.body_offset.value
            )
            self.user_params.body_offset.changed = False
        # Swimming
        if self.gait == "swimming":
            self.forces_torques[self.sim_step] = viscous_swimming(
                self.salamander.identity,
                self.links
            )
        # Time plugins
        self.time_plugin = time.time() - self.tic_rt
        self.tot_plugin_time += self.time_plugin
        # Control
        self.tic_control = time.time()
        self.model.controller.control()
        self.time_control = time.time() - self.tic_control
        self.tot_ctrl_time += self.time_control
        # Physics
        self.tic_sim = time.time()
        pybullet.stepSimulation()
        self.sim_step += 1
        self.toc_sim = time.time()
        self.tot_sim_time += self.toc_sim - self.tic_sim
        # Contacts during walking
        tic_sensors = time.time()
        self.salamander.sensors.update(
            identity=self.salamander.identity,
            links=[self.links[foot] for foot in self.salamander.feet],
            joints=[
                self.joints[joint]
                for joint in self.salamander.sensors.joints_sensors
            ],
            plane=self.plane
        )
        # Commands
        self.salamander.motors.update(
            identity=self.salamander.identity,
            joints_body=[
                self.joints[joint]
                for joint in self.salamander.motors.joints_commanded_body
            ],
            joints_legs=[
                self.joints[joint]
                for joint in self.salamander.motors.joints_commanded_legs
            ]
        )
        time_sensors = time.time() - tic_sensors
        self.tot_sensors_time += time_sensors
        tic_log = time.time()
        self.experiment_logger.update(self.sim_step-1)
        time_log = time.time() - tic_log
        self.tot_log_time += time_log
        # Camera
        tic_camera = time.time()
        if clargs.record and not self.sim_step % 25:
            self.camera_record.record(self.sim_step//25-1)
        # User camera
        if not self.sim_step % 10 and not clargs.free_camera:
            self.camera.update()
        self.tot_camera_time += time.time() - tic_camera
        # Real-time
        self.toc_rt = time.time()
        tic_rt = time.time()
        if not clargs.fast and self.user_params.rtl.value < 3:
            real_time_handing(
                self.timestep, self.tic_rt, self.toc_rt,
                rtl=self.user_params.rtl.value,
                time_plugin=self.time_plugin,
                time_sim=self.toc_sim-self.tic_sim,
                time_control=self.time_control
            )
        self.tot_waitrt_time = time.time() - tic_rt

    def end(self, clargs):
        """Terminate simulation"""
        # Simulation information
        self.sim_time = self.timestep*self.sim_step
        print("Time to simulate {} [s]: {} [s]".format(
            self.sim_time,
            self.toc-self.tic,
        ))
        print("  Plugin: {} [s]".format(self.tot_plugin_time))
        print("  Bullet physics: {} [s]".format(self.tot_sim_time))
        print("  Controller: {} [s]".format(self.tot_ctrl_time))
        print("  Sensors: {} [s]".format(self.tot_sensors_time))
        print("  Logging: {} [s]".format(self.tot_log_time))
        print("  Camera: {} [s]".format(self.tot_camera_time))
        print("  Wait real-time: {} [s]".format(self.tot_waitrt_time))
        print("  Sum: {} [s]".format(
            self.tot_plugin_time
            + self.tot_sim_time
            + self.tot_ctrl_time
            + self.tot_sensors_time
            + self.tot_log_time
            + self.tot_camera_time
            + self.tot_waitrt_time
        ))

        # Disconnect from simulation
        pybullet.disconnect()

        # Plot
        self.experiment_logger.plot_all(self.times_simulated)
        plt.show()

        # Record video
        if clargs.record:
            self.camera_record.save("video.avi")


class Model:
    """Simulation model"""

    def __init__(self, identity, base_link="base_link"):
        super(Model, self).__init__()
        self.identity = identity
        self.links, self.joints, self.n_joints = self.get_joints(
            self.identity,
            base_link
        )
        self.print_information()

    @classmethod
    def from_sdf(cls, sdf, base_link="base_link", **kwargs):
        """Model from SDF"""
        identity = pybullet.loadSDF(sdf)[0]
        return cls(identity, base_link=base_link, **kwargs)

    @classmethod
    def from_urdf(cls, urdf, base_link="base_link", **kwargs):
        """Model from SDF"""
        identity = pybullet.loadURDF(urdf, **kwargs)
        return cls(identity, base_link=base_link)

    @staticmethod
    def get_joints(identity, base_link="base_link"):
        """Get joints"""
        print("Identity: {}".format(identity))
        n_joints = pybullet.getNumJoints(identity)
        print("Number of joints: {}".format(n_joints))

        # Links
        # Base link
        links = {base_link: -1}
        links.update({
            info[12].decode("UTF-8"): info[16] + 1
            for info in [
                pybullet.getJointInfo(identity, j)
                for j in range(n_joints)
            ]
        })
        # Joints
        joints = {
            info[1].decode("UTF-8"): info[0]
            for info in [
                pybullet.getJointInfo(identity, j)
                for j in range(n_joints)
            ]
        }
        return links, joints, n_joints

    def print_information(self):
        """Print information"""
        print("Links ids:\n{}".format(
            "\n".join([
                "  {}: {}".format(
                    name,
                    self.links[name]
                )
                for name in self.links
            ])
        ))
        print("Joints ids:\n{}".format(
            "\n".join([
                "  {}: {}".format(
                    name,
                    self.joints[name]
                )
                for name in self.joints
            ])
        ))

    def print_dynamics_info(self, links=None):
        """Print dynamics info"""
        links = links if links is not None else self.links
        print("Dynamics:")
        for link in links:
            dynamics_msg = (
                "\n      mass: {}"
                "\n      lateral_friction: {}"
                "\n      local inertia diagonal: {}"
                "\n      local inertial pos: {}"
                "\n      local inertial orn: {}"
                "\n      restitution: {}"
                "\n      rolling friction: {}"
                "\n      spinning friction: {}"
                "\n      contact damping: {}"
                "\n      contact stiffness: {}"
            )

            print("  - {}:{}".format(
                link,
                dynamics_msg.format(*pybullet.getDynamicsInfo(
                    self.identity,
                    self.links[link]
                ))
            ))
        print("Model mass: {} [kg]".format(self.mass()))

    def mass(self):
        """Print dynamics"""
        return np.sum([
            pybullet.getDynamicsInfo(self.identity, self.links[link])[0]
            for link in self.links
        ])


class SalamanderModel(Model):
    """Salamander model"""

    def __init__(self, identity, base_link, timestep, gait="walking"):
        super(SalamanderModel, self).__init__(
            identity=identity,
            base_link=base_link
        )
        # Model dynamics
        self.apply_motor_damping()
        # Controller
        self.controller = SalamanderController.from_gait(
            self.identity,
            self.joints,
            gait=gait,
            timestep=timestep
        )
        self.feet = [
            "link_leg_0_L_3",
            "link_leg_0_R_3",
            "link_leg_1_L_3",
            "link_leg_1_R_3"
        ]
        self.sensors = ModelSensors(self)
        self.motors = ModelMotors()

    @classmethod
    def spawn(cls, timestep, gait="walking"):
        """Spawn salamander"""
        return cls.from_sdf(
            "/home/jonathan/.gazebo/models/biorob_salamander/model.sdf",
            base_link="link_body_0",
            timestep=timestep,
            gait=gait
        )

    def leg_collisions(self, plane, activate=True):
        """Activate/Deactivate leg collisions"""
        for leg_i in range(2):
            for side in ["L", "R"]:
                for joint_i in range(3):
                    link = "link_leg_{}_{}_{}".format(leg_i, side, joint_i)
                    pybullet.setCollisionFilterPair(
                        bodyUniqueIdA=self.identity,
                        bodyUniqueIdB=plane,
                        linkIndexA=self.links[link],
                        linkIndexB=-1,
                        enableCollision=activate
                    )

    def apply_motor_damping(self, linear=0, angular=0):
        """Apply motor damping"""
        for j in range(pybullet.getNumJoints(self.identity)):
            pybullet.changeDynamics(
                self.identity, j,
                linearDamping=0,
                angularDamping=angular
            )


class ModelSensors:
    """Model sensors"""

    def __init__(self, salamander):  # , sensors
        super(ModelSensors, self).__init__()
        # self.sensors = sensors
        # Contact sensors
        self.feet = salamander.feet
        self.contact_forces = np.zeros([4])

        # Force-torque sensors
        self.feet_ft = np.zeros([4, 6])
        self.joints_sensors = [
            "joint_link_leg_0_L_3",
            "joint_link_leg_0_R_3",
            "joint_link_leg_1_L_3",
            "joint_link_leg_1_R_3"
        ]
        for joint in self.joints_sensors:
            pybullet.enableJointForceTorqueSensor(
                salamander.identity,
                salamander.joints[joint]
            )

    def update(self, identity, links, joints, plane):
        """Update sensors"""
        self.update_contacts(identity, links, plane)
        self.update_joints(identity, joints)

    def update_contacts(self, identity, links, plane):
        """Update contact sensors"""
        _, self.contact_forces = (
            self.get_links_contacts(identity, links, plane)
        )

    def update_joints(self, identity, joints):
        """Update force-torque sensors"""
        self.feet_ft = (
            self.get_joints_force_torque(identity, joints)
        )

    @staticmethod
    def get_links_contacts(identity, links, ground):
        """Contacts"""
        contacts = [
            pybullet.getContactPoints(identity, ground, link, -1)
            for link in links
        ]
        forces = [
            np.sum([contact[9] for contact in contacts[link_i]])
            if contacts
            else 0
            for link_i, _ in enumerate(links)
        ]
        return contacts, forces

    @staticmethod
    def get_joints_force_torque(identity, joints):
        """Force-torque on joints"""
        return [
            pybullet.getJointState(identity, joint)[2]
            for joint in joints
        ]


class ModelMotors:
    """Model motors"""

    def __init__(self):
        super(ModelMotors, self).__init__()
        # Commands
        self.joints_commanded_body = [
            "joint_link_body_{}".format(joint_i+1)
            for joint_i in range(11)
        ]
        self.joints_commanded_legs = [
            "joint_link_leg_{}_{}_{}".format(leg_i, side, joint_i)
            for leg_i in range(2)
            for side in ["L", "R"]
            for joint_i in range(3)
        ]
        self.joints_cmds_body = np.zeros(len(self.joints_commanded_body))
        self.joints_cmds_legs = np.zeros(len(self.joints_commanded_legs))

    def update(self, identity, joints_body, joints_legs):
        """Update"""
        self.update_body(identity, joints_body)
        self.update_legs(identity, joints_legs)

    def update_body(self, identity, joints):
        """Update"""
        self.joints_cmds_body = (
            self.get_joints_commands(identity, joints)
        )

    def update_legs(self, identity, joints):
        """Update"""
        self.joints_cmds_legs = (
            self.get_joints_commands(identity, joints)
        )

    @staticmethod
    def get_joints_commands(identity, joints):
        """Force-torque on joints"""
        return [
            pybullet.getJointState(identity, joint)[3]
            for joint in joints
        ]


class Camera:
    """Camera"""

    def __init__(self, timestep, target_identity=None, **kwargs):
        super(Camera, self).__init__()
        self.target = target_identity
        cam_info = self.get_camera()
        self.timestep = timestep
        self.motion_filter = kwargs.pop("motion_filter", 1e-2)
        self.yaw = kwargs.pop("yaw", cam_info[8])
        self.yaw_speed = kwargs.pop("yaw_speed", 0)
        self.pitch = kwargs.pop("pitch", cam_info[9])
        self.distance = kwargs.pop("distance", cam_info[10])

    @staticmethod
    def get_camera():
        """Get camera information"""
        return pybullet.getDebugVisualizerCamera()

    def update_yaw(self):
        """Update yaw"""
        self.yaw += self.yaw_speed*self.timestep


class CameraTarget(Camera):
    """Camera with target following"""

    def __init__(self, target_identity, **kwargs):
        super(CameraTarget, self).__init__(**kwargs)
        self.target = target_identity
        self.target_pos = kwargs.pop(
            "target_pos",
            np.array(pybullet.getBasePositionAndOrientation(self.target)[0])
            if self.target is not None
            else np.array(self.get_camera()[11])
        )

    def update_target_pos(self):
        """Update target position"""
        self.target_pos = (
            (1-self.motion_filter)*self.target_pos
            + self.motion_filter*np.array(
                pybullet.getBasePositionAndOrientation(self.target)[0]
            )
        )


class UserCamera(CameraTarget):
    """UserCamera"""

    def __init__(self, target_identity, **kwargs):
        super(UserCamera, self).__init__(target_identity, **kwargs)
        self.update(use_camera=False)

    def update(self, use_camera=True):
        """Camera view"""
        if use_camera:
            self.yaw, self.pitch, self.distance = self.get_camera()[8:11]
        self.update_yaw()
        self.update_target_pos()
        pybullet.resetDebugVisualizerCamera(
            cameraDistance=self.distance,
            cameraYaw=self.yaw,
            cameraPitch=self.pitch,
            cameraTargetPosition=self.target_pos
        )


class CameraRecord(CameraTarget):
    """Camera recording"""

    def __init__(self, target_identity, size, fps, **kwargs):
        super(CameraRecord, self).__init__(target_identity, **kwargs)
        self.width = kwargs.pop("width", 640)
        self.height = kwargs.pop("height", 480)
        self.fps = fps
        self.data = np.zeros(
            [size, self.height, self.width, 4],
            dtype=np.uint8
        )

    def record(self, sample):
        """Record camera"""
        self.update_yaw()
        self.update_target_pos()
        self.data[sample, :, :] = pybullet.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=pybullet.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.target_pos,
                distance=self.distance,
                yaw=self.yaw,
                pitch=self.pitch,
                roll=0,
                upAxisIndex=2
            ),
            projectionMatrix = pybullet.computeProjectionMatrixFOV(
                fov=60,
                aspect=640/480,
                nearVal=0.1,
                farVal=5
            ),
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
            flags=pybullet.ER_NO_SEGMENTATION_MASK
        )[2]

    def save(self, filename="video.avi"):
        """Save recording"""
        print("Recording video to {}".format(filename))
        import cv2
        writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*'MJPG'),
            self.fps,
            (self.width, self.height)
        )
        for image in self.data:
            writer.write(image)


class DebugParameter:
    """DebugParameter"""

    def __init__(self, name, val, val_min, val_max):
        super(DebugParameter, self).__init__()
        self.name = name
        self.value = val
        self.val_min = val_min
        self.val_max = val_max
        self._handler = None
        self.add(self.value)

    def add(self, value):
        """Add parameter"""
        if self._handler is None:
            self._handler = pybullet.addUserDebugParameter(
                paramName=self.name,
                rangeMin=self.val_min,
                rangeMax=self.val_max,
                startValue=value
            )
        else:
            raise Exception(
                "Handler for parameter '{}' is already used".format(
                    self.name
                )
            )

    def remove(self):
        """Remove parameter"""
        pybullet.removeUserDebugItem(self._handler)

    def get_value(self):
        """Current value"""
        return pybullet.readUserDebugParameter(self._handler)


class ParameterPlay(DebugParameter):
    """Play/pause parameter"""

    def __init__(self):
        super(ParameterPlay, self).__init__("Play", 1, 0, 1)
        self.value = True

    def update(self):
        """Update"""
        self.value = self.get_value() > 0.5


class ParameterRTL(DebugParameter):
    """Real-time limiter"""

    def __init__(self):
        super(ParameterRTL, self).__init__("Real-time limiter", 1, 1e-3, 3)

    def update(self):
        """Update"""
        self.value = self.get_value()


class ParameterGait(DebugParameter):
    """Gait control"""

    def __init__(self, gait):
        value = 0 if gait == "standing" else 2 if gait == "swimming" else 1
        super(ParameterGait, self).__init__("Gait", value, 0, 2)
        self.value = gait
        self.changed = False

    def update(self):
        """Update"""
        previous_value = self.value
        value = self.get_value()
        self.value = (
            "standing"
            if value < 0.5
            else "walking"
            if 0.5 < value < 1.5
            else "swimming"
        )
        self.changed = (self.value != previous_value)
        if self.changed:
            print("Gait changed ({} > {})".format(
                previous_value,
                self.value
            ))


class ParameterFrequency(DebugParameter):
    """Frequency control"""

    def __init__(self, frequency):
        super(ParameterFrequency, self).__init__("Frequency", frequency, 0, 5)
        self.changed = False

    def update(self):
        """Update"""
        previous_value = self.value
        self.value = self.get_value()
        self.changed = (self.value != previous_value)
        if self.changed:
            print("frequency changed ({} > {})".format(
                previous_value,
                self.value
            ))


class ParameterBodyOffset(DebugParameter):
    """Body offset control"""

    def __init__(self):
        lim = np.pi/8
        super(ParameterBodyOffset, self).__init__("Body offset", 0, -lim, lim)
        self.changed = False

    def update(self):
        """Update"""
        previous_value = self.value
        self.value = self.get_value()
        self.changed = (self.value != previous_value)
        if self.changed:
            print("Body offset changed ({} > {})".format(
                previous_value,
                self.value
            ))


class UserParameters:
    """Parameters control"""

    def __init__(self, gait, frequency):
        super(UserParameters, self).__init__()
        self._play = ParameterPlay()
        self._rtl = ParameterRTL()
        self._gait = ParameterGait(gait)
        self._frequency = ParameterFrequency(frequency)
        self._body_offset = ParameterBodyOffset()

    def update(self):
        """Update parameters"""
        for parameter in [
                self._play,
                self._rtl,
                self._gait,
                self._frequency,
                self._body_offset
        ]:
            parameter.update()

    @property
    def play(self):
        """Play"""
        return self._play

    @property
    def rtl(self):
        """Real-time limiter"""
        return self._rtl

    @property
    def gait(self):
        """Gait"""
        return self._gait

    @property
    def frequency(self):
        """Frequency"""
        return self._frequency

    @property
    def body_offset(self):
        """Body offset"""
        return self._body_offset


def rendering(render=1):
    """Enable/disable rendering"""
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, render)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, render)
    # pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_TINY_RENDERER, render)


class ExperimentLogger:
    """ExperimentLogger"""

    def __init__(self, model, sim_size):
        super(ExperimentLogger, self).__init__()
        self.sim_size = sim_size
        self.model = model
        self.sensors = SensorsLogger(model, sim_size)
        # [SensorsLogger(model) for sensor in model.sensors]
        self.motors = MotorsLogger(model, sim_size)
        self.phases = PhasesLogger(model, sim_size)

    def update(self, iteration):
        """Update sensors at iteration"""
        self.sensors.update(iteration)
        self.motors.update(iteration)
        self.phases.update(iteration)

    def plot_all(self, sim_times):
        """Plot all"""
        self.sensors.plot_contacts(sim_times)
        self.sensors.plot_ft(sim_times)
        self.motors.plot_body(sim_times)
        self.motors.plot_legs(sim_times)
        self.phases.plot(sim_times)


class SensorsLogger:
    """Sensors logger"""

    def __init__(self, model, size):
        super(SensorsLogger, self).__init__()
        self.model = model
        self.size = size
        self.contact_forces = np.zeros([
            size,
            *np.shape(model.sensors.contact_forces)
        ])
        self.feet_ft = np.zeros([
            size,
            *np.shape(model.sensors.feet_ft)
        ])
        self.feet = model.sensors.feet

    def update(self, iteration):
        """Update sensors logs"""
        self.contact_forces[iteration, :] = self.model.sensors.contact_forces
        self.feet_ft[iteration, :, :] = self.model.sensors.feet_ft

    def plot_contacts(self, times):
        """Plot sensors"""
        # Plot contacts
        plt.figure("Contacts")
        for foot_i, foot in enumerate(self.feet):
            plt.plot(
                times,
                self.contact_forces[:len(times), foot_i],
                label=foot
            )
            plt.xlabel("Time [s]")
            plt.ylabel("Reaction force [N]")
            plt.grid(True)
            plt.legend()

    def plot_ft(self, times):
        """Plot force-torque sensors"""
        # Plot Feet forces
        plt.figure("Feet forces")
        for dim in range(3):
            plt.plot(
                times,
                self.feet_ft[:len(times), 0, dim],
                label=["x", "y", "z"][dim]
            )
            plt.xlabel("Time [s]")
            plt.ylabel("Force [N]")
            plt.grid(True)
            plt.legend()


class MotorsLogger:
    """Motors logger"""

    def __init__(self, model, size):
        super(MotorsLogger, self).__init__()
        self.model = model
        self.size = size
        self.joints_cmds_body = np.zeros([
            size,
            *np.shape(model.motors.joints_cmds_body)
        ])
        self.joints_commanded_body = model.motors.joints_commanded_body
        self.joints_cmds_legs = np.zeros([
            size,
            *np.shape(model.motors.joints_cmds_legs)
        ])
        self.joints_commanded_legs = model.motors.joints_commanded_legs

    def update(self, iteration):
        """Update motor logs"""
        self.joints_cmds_body[iteration, :] = (
            self.model.motors.joints_cmds_body
        )
        self.joints_cmds_legs[iteration, :] = (
            self.model.motors.joints_cmds_legs
        )

    def plot_body(self, times):
        """Plot body motors"""
        plt.figure("Body motor torques")
        for joint_i, joint in enumerate(self.joints_commanded_body):
            plt.plot(
                times,
                self.joints_cmds_body[:len(times), joint_i],
                label=joint
            )
            plt.xlabel("Time [s]")
            plt.ylabel("Torque [Nm]")
            plt.grid(True)
            plt.legend()

    def plot_legs(self, times):
        """Plot legs motors"""
        plt.figure("Legs motor torques")
        for joint_i, joint in enumerate(self.joints_commanded_legs):
            plt.plot(
                times,
                self.joints_cmds_legs[:len(times), joint_i],
                label=joint
            )
            plt.xlabel("Time [s]")
            plt.ylabel("Torque [Nm]")
            plt.grid(True)
            plt.legend()


class PhasesLogger:
    """Phases logger"""

    def __init__(self, model, size):
        super(PhasesLogger, self).__init__()
        self.model = model
        self.size = size
        self.phases_log = np.zeros([
            size,
            *np.shape(model.controller.network.phases)
        ])
        self.oscillator_names = [
            "body_{}".format(i)
            for i in range(11)
        ] +  [
            "leg_{}_{}_{}".format(leg_i, side, joint_i)
            for leg_i in range(2)
            for side in ["L", "R"]
            for joint_i in range(3)
        ]

    def update(self, iteration):
        """Update phase logs"""
        self.phases_log[iteration, :] = (
            self.model.controller.network.phases[:, 0]
        )

    def plot(self, times):
        """Plot body phases"""

        for phase_i, phase in enumerate(self.oscillator_names):
            if "body" in phase:
                plt.figure("Oscillator body phases")
            else:
                plt.figure("Oscillator legs phases")
            plt.plot(
                times,
                self.phases_log[:len(times), phase_i],
                label=phase
            )
            plt.xlabel("Time [s]")
            plt.ylabel("Phase [rad]")
            plt.grid(True)
            plt.legend()


def main(clargs=None):
    """Main"""

    # Parse command line arguments
    if not clargs:
        clargs = parse_args()

    # Setup simulation
    sim = Simulation(
        timestep=1e-3,
        duration=10,
        clargs=clargs,
        gait="walking"
    )

    # Run simulation
    sim.run(clargs)

    # Show results
    sim.end(clargs)


def main_parallel():
    """Simulation with multiprocessing"""
    from multiprocessing import Pool

    # Parse command line arguments
    clargs = parse_args()

    # Create Pool
    pool = Pool(2)

    # Run simulation
    pool.map(main, [clargs, clargs])
    print("Done")


if __name__ == '__main__':
    # main_parallel()
    main()
