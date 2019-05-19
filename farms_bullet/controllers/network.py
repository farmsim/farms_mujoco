"""Network"""

import numpy as np
from ..cy_controller import ode_oscillators_sparse, rk4
from .convention import bodyosc2index, legosc2index, legjoint2index
from .control_options import SalamanderControlOptions
from ..animats.model_options2 import ModelOptions
import pdb

class ODE(list):
    """ODE"""

    def __init__(self, solver, function):
        super(ODE, self).__init__([solver, function])

    @property
    def solver(self):
        """Solver"""
        return self[0]

    @property
    def function(self):
        """Function"""
        return self[1]


class CyODESolver:
    """ODE solver"""

    def __init__(self, ode, state, timestep, parameters, **kwargs):
        super(CyODESolver, self).__init__()
        self.ode = ode
        self._state = state
        self._n_dim = np.shape(state)[2]
        self._timestep = timestep
        self._iteration = kwargs.pop("iteration", 0)
        self._parameters = parameters

    @property
    def current_state(self):
        """State"""
        return self._state[self._iteration, 0]

    @property
    def current_dstate(self):
        """State derivative"""
        return self._state[self._iteration, 1]

    @property
    def iteration(self):
        """Iteration"""
        return self._iteration

    def step(self):
        """Control step"""
        self.ode.solver(
            self.ode.function,
            self._timestep,
            self._state,
            self._n_dim,
            self._iteration,
            *self._parameters.solver,
            self._parameters.function
        )
        self._iteration += 1


class ODESolver(CyODESolver):
    """ODE solver over time"""

    def __init__(self, ode, state, timestep, **kwargs):
        super(ODESolver, self).__init__(ode, state, timestep, **kwargs)
        iterations = np.shape(state)[0]
        self._times = np.arange(0, timestep * iterations, timestep)
        assert len(self._times) == iterations

    @property
    def time(self):
        """Time"""
        return self._times[self._iteration]


class NetworkArray:
    """Network array"""

    def __init__(self, array):
        super(NetworkArray, self).__init__()
        self._array = array

    @property
    def array(self):
        """Array"""
        return self._array

    def shape(self):
        """Array shape"""
        return np.shape(self._array)


class OscillatorNetworkState(NetworkArray):
    """Network state"""

    def __init__(self, state, n_oscillators, iteration=0):
        self.n_oscillators = n_oscillators
        self._iteration = iteration
        super(OscillatorNetworkState, self).__init__(state)

    @staticmethod
    def default_initial_state():
        """Default state"""
        n_joints = 11 + 4 * 3
        return np.linspace(0, 1e-6, 5 * n_joints)

    @staticmethod
    def default_state(n_iterations):
        """Default state"""
        n_joints = 11 + 4 * 3
        n_oscillators = 2 * n_joints
        return OscillatorNetworkState.from_initial_state(
            initial_state=OscillatorNetworkState.default_initial_state(),
            n_iterations=n_iterations,
            n_oscillators=n_oscillators
        )

    @classmethod
    def from_initial_state(cls, initial_state, n_iterations, n_oscillators):
        """From initial state"""
        state = np.zeros(
            [n_iterations, 2, np.shape(initial_state)[0]],
            dtype=np.float64
        )
        state[0, 0] = np.array(initial_state)
        return cls(state, n_oscillators)

    @classmethod
    def from_solver(cls, solver, n_oscillators):
        """From solver"""
        return cls(solver.state, n_oscillators, solver.iteration)

    def phases(self, iteration):
        """Phases"""
        return self.array[iteration, 0, :self.n_oscillators]

    def amplitudes(self, iteration):
        """Amplitudes"""
        return self.array[iteration, 0, self.n_oscillators:]

    def dphases(self, iteration):
        """Phases derivative"""
        return self.array[iteration, 1, :self.n_oscillators]

    def damplitudes(self, iteration):
        """Amplitudes derivative"""
        return self.array[iteration, 1, self.n_oscillators:]


class SalamanderNetworkParameters(ODE):
    """Salamander network parameter"""

    def __init__(self, oscillators, connectivity, joints):
        super(SalamanderNetworkParameters, self).__init__(
            [NetworkArray(np.zeros([
                7,
                2 * oscillators.shape()[1] + 1 * joints.shape()[1]
            ]))],
            [oscillators, connectivity, joints]
        )
        

    @classmethod
    def from_gait(cls, gait):
        """ Salamander network parameters from gait"""
        return (
            cls.for_swimming()
            if gait == "swimming"
            else cls.for_walking()
        )

    def update_gait(self, gait):
        """Update from gait"""
        self[1][0] = OscillatorArray.for_moving()
        self[1][1] = ConnectivityArray.for_moving()
        self[1][2] = JointsArray.for_moving()
        
    @staticmethod
    def my_network():
        """implementation of blaise"""    
        oscillators = OscillatorArray.for_moving()
        connectivity = ConnectivityArray.for_moving()
        joints = JointsArray.for_moving()
        #pdb.set_trace()
        return oscillators, connectivity, joints

    # @staticmethod
    # def walking_parameters():
    #     """Walking parameters"""
    #     raise Exception
    #     oscillators = OscillatorArray.for_walking()
    #     connectivity = ConnectivityArray.for_walking()
    #     joints = JointsArray.for_walking()
    #     return oscillators, connectivity, joints

    # @staticmethod
    # def swimming_parameters():
    #     """Swimming parameters"""
    #     raise Exception
    #     oscillators = OscillatorArray.for_swimming()
    #     connectivity = ConnectivityArray.for_swimming()
    #     joints = JointsArray.for_swimming()
    #     return oscillators, connectivity, joints

    @classmethod
    def for_walking(cls):
        """Salamander swimming network"""
        oscillators, connectivity, joints = cls.my_network()
        return cls(oscillators, connectivity, joints)

    @classmethod
    def for_swimming(cls):
        """Salamander swimming network"""
        oscillators, connectivity, joints = cls.my_networks()
        return cls(oscillators, connectivity, joints)

    @property
    def rk4(self):
        """Rung-Kutta parameters"""
        return self[0][0]

    @property
    def oscillators(self):
        """Oscillators parameters"""
        return self[1][0]

    @property
    def connectivity(self):
        """Connectivity parameters"""
        return self[1][1]

    @property
    def joints(self):
        """Joints parameters"""
        return self[1][2]

    def to_ode_parameters(self):
        """Convert 2 arrays"""
        return ODE(
            [parameter.array for parameter in self.solver],
            [parameter.array for parameter in self.function]
            + [self.oscillators.shape()[1]]
            + [self.connectivity.shape()[0]]
            + [self.joints.shape()[1]]
        )


class  OscillatorArray(NetworkArray):
    """Oscillator array"""

    def __init__(self, array):
        super(OscillatorArray, self).__init__(array)
        self._array = array
        self._original_amplitudes_desired = np.copy(array[2])
        self._opt = ModelOptions()
        #my implementation
        

    @classmethod
    def from_parameters(cls, freqs, rates, amplitudes):
        """From each parameter"""
        return cls(np.array([freqs, rates, amplitudes]))

    @staticmethod
    def load_params():
        """my implementation"""
        _options = ModelOptions()
        #freqs
        body_freqs = 2 * np.pi * _options['body_freqs'] * np.ones(2 * _options['n_body']) 
        limb_freqs = 2 * np.pi * _options['limb_freqs'] * np.ones(2 * _options["n_dof_legs"] * _options['n_legs'])
        freqs = np.append(body_freqs, limb_freqs)
        #rates
        rates = _options['rates'] * np.ones(np.shape(freqs)[0])
        #amplitudes
        body_amp = _options['body_amp'] * 2
        forelimb_amp = np.append(_options['left_forelimb_amp'], _options['right_forelimb_amp'])
        hindlimb_amp = np.append(_options['left_hindlimb_amp'], _options['right_hindlimb_amp'])
        limb_amp = np.append(forelimb_amp, hindlimb_amp)
        amplitudes = np.append(body_amp, limb_amp)

        debug = True
        if debug == True:
            print("-----freqs-----")
            print(freqs)
            print("-----rates-----")
            print(rates)
            print("-----amplitudes-----")
            print(amplitudes)
            
        return freqs, rates, amplitudes

    # @staticmethod
    # def walking_parameters():
    #     """Walking parameters"""
    #     raise Exception

    #     opt_mod = ModelOptions()
    #     n_body = opt_mod['n_body']
    #     n_dof_legs = opt_mod['n_dof_legs']
    #     n_legs = opt_mod['n_legs']
    #     n_joints = n_body + n_legs * n_dof_legs
    #     n_oscillators = 2 * (n_joints)
    #     freqs = 2 * np.pi * np.ones(n_oscillators)
    #     rates = 10 * np.ones(n_oscillators)
    #     options = SalamanderControlOptions.walking()
    #     # Amplitudes
    #     amplitudes = np.zeros(n_oscillators)
    #     for i in range(n_body):
    #         amplitudes[[i, i + n_body]] = np.abs(
    #             options["body_stand_amplitude"] * np.sin(
    #                 2 * np.pi * i / n_body
    #                 - options["body_stand_shift"]
    #             )
    #         )
    #     for leg_i in range(n_legs):
    #         for i in range(n_dof_legs):
    #             amplitudes[[
    #                 2 * n_body + 2 * leg_i * n_dof_legs + i,
    #                 2 * n_body + 2 * leg_i * n_dof_legs + i + n_dof_legs
    #             ]] = np.abs(
    #                 options["leg_{}_amplitude".format(i)]
    #             )
    #     return freqs, rates, amplitudes

    # @staticmethod
    # def swimming_parameters():
    #     """Swimming parameters"""
    #     raise Exception

    #     opt_mod = ModelOptions()
    #     n_body = opt_mod['n_body']
    #     n_dof_legs = opt_mod['n_dof_legs']
    #     n_legs = opt_mod['n_legs']
    #     n_joints = n_body + n_legs * n_dof_legs
    #     n_oscillators = 2 * (n_joints)
    #     freqs = 2 * np.pi * np.ones(n_oscillators)
    #     rates = 10 * np.ones(n_oscillators)
    #     amplitudes = np.zeros(n_oscillators)
    #     options = SalamanderControlOptions.swimming()
    #     body_amplitudes = np.linspace(
    #         options["body_amplitude_0"],
    #         options["body_amplitude_1"],
    #         n_body
    #     )
    #     for i in range(n_body):
    #         amplitudes[[i, i + n_body]] = body_amplitudes[i]
    #     for leg_i in range(n_legs):
    #         for i in range(n_dof_legs):
    #             amplitudes[[
    #                 2 * n_body + 2 * leg_i * n_dof_legs + i,
    #                 2 * n_body + 2 * leg_i * n_dof_legs + i + n_dof_legs
    #             ]] = (
    #                 options["leg_{}_amplitude".format(i)]
    #             )
    #     return freqs, rates, amplitudes

    @classmethod
    def for_moving(cls):
        freqs, rates, amplitudes = cls.load_params()    
        return cls.from_parameters(freqs, rates, amplitudes)

    @classmethod
    def for_walking(cls):
        """Parameters for walking"""
        freqs, rates, amplitudes = cls.load_params()
        #freqs, rates, amplitudes = cls.walking_parameters()
        #pdb.set_trace()
        return cls.from_parameters(freqs, rates, amplitudes)

    @classmethod
    def for_swimming(cls):
        """Parameters for swimming"""
        freqs, rates, amplitudes = cls.load_params()
        return cls.from_parameters(freqs, rates, amplitudes)

    @property
    def freqs(self):
        """Frequencies"""
        return self.array[0]

    @freqs.setter
    def freqs(self, value):
        """Frequencies"""
        self.array[0, :] = value

    @property
    def amplitudes_rates(self):
        """Amplitudes rates"""
        return self.array[1]

    @property
    def amplitudes_desired(self):
        """Amplitudes desired"""
        return self.array[2]

    @amplitudes_desired.setter
    def amplitudes_desired(self, value):
        """Amplitudes desired"""
        self.array[2, :] = value

    def freq_sat_body(self, drive_speed, drive_turn):
        drive_low_sat = 1
        drive_up_sat = 5
        dim_body = 22
        if drive_low_sat <= drive_speed <= drive_up_sat:
            self.freqs[:dim_body] = 2 * np.pi * (1 + 0.4 * drive_speed)
        else:
            self.freqs[:dim_body] = 0

    def freq_sat_limb(self, drive_speed, drive_turn):
        """
        :param drive_speed:
        :return:
        """
        drive_low_sat = 1
        drive_up_sat = 3
        dim_body = 22
        if drive_low_sat <= drive_speed <= drive_up_sat:
            self.freqs[dim_body:] = 2 * np.pi * (0.5 + 0.2 * drive_speed)
        else:
            self.freqs[dim_body:] = 0

    def amp_sat_body(self, drive_speed, drive_turn):
        """
        :param drive_speed:
        :return:
        """
        drive_low_sat = 1
        drive_up_sat = 5
        dim_body = 22
        if drive_low_sat <= drive_speed <= drive_up_sat:
            self.amplitudes_desired[:dim_body] = np.append(np.linspace(0.1, 0.5, 11)/2,np.linspace(0.1, 0.5, 11)/2)  + 0.04 * drive_speed
        else:
            self.amplitudes_desired[:dim_body] = 0

    def amp_sat_limb(self, drive_speed, drive_turn):
        """
        function that saturated the 3 DOFs of each limb, the shoulder has 2 DOFs and 1 for the KNEE
        :param drive_speed:
        :return: the saturation of the
        """

        drive_low_sat = 1
        drive_up_sat = 3
        dim_body = 22
        forelimb_amp = np.append(self._opt['left_forelimb_amp'], self._opt['right_forelimb_amp'])
        hindlimb_amp = np.append(self._opt['left_hindlimb_amp'], self._opt['right_hindlimb_amp'])
        limb_amp = np.append(forelimb_amp, hindlimb_amp)

        if drive_low_sat <= drive_speed <= drive_up_sat:
            # # characterizing the elbow amplitudes
            for i in np.arange(2):
                for j in np.arange(2):
                     # forward motion of the shoulder 0-90
                    self.amplitudes_desired[dim_body:] = limb_amp+0.05*(drive_speed - drive_low_sat)
            
                    print('====update amplitudes walking=====')
                    print(self.amplitudes_desired)

        else:
            self.amplitudes_desired[dim_body:] = 0
            print('====update amplitudes swimming=====')
            print(self.amplitudes_desired)

    def update_drives(self, drive_speed, drive_turn):
        """
        :param drive_speed: drive that change the frequency
        :param drive_turn: drive that change the offset
        :return: send to the simulation the drive
        """
        self.freq_sat_limb(drive_speed, drive_turn)
        self.freq_sat_body(drive_speed, drive_turn)
        self.amp_sat_body(drive_speed, drive_turn)
        self.amp_sat_limb(drive_speed, drive_turn)


class ConnectivityArray(NetworkArray):
    """Connectivity array"""

    @classmethod
    def from_parameters(cls, connections, weights, desired_phases):
        """From each parameter"""
        return cls(np.stack([connections, weights, desired_phases], axis=1))

    # @staticmethod
    # def walking_parameters():
    #     """Walking parameters"""
    #     raise Exception
    #     n_body_joints = 11
    #     connectivity = []
    #     default_amplitude = 3e2
    #
    #     # Amplitudes
    #     options = SalamanderControlOptions.walking()
    #     amplitudes = [
    #         options["body_stand_amplitude"] * np.sin(
    #             2 * np.pi * i / n_body_joints
    #             - options["body_stand_shift"]
    #         )
    #         for i in range(n_body_joints)
    #     ]
    #
    #     # Body
    #     for i in range(n_body_joints - 1):
    #         # i - i
    #         connectivity.append([
    #             bodyosc2index(joint_i=i, side=1),
    #             bodyosc2index(joint_i=i, side=0),
    #             default_amplitude, np.pi
    #         ])
    #         connectivity.append([
    #             bodyosc2index(joint_i=i, side=0),
    #             bodyosc2index(joint_i=i, side=1),
    #             default_amplitude, np.pi
    #         ])
    #         # i - i+1
    #         phase_diff = (
    #             0
    #             if np.sign(amplitudes[i]) == np.sign(amplitudes[i + 1])
    #             else np.pi
    #         )
    #         for side in range(2):
    #             connectivity.append([
    #                 bodyosc2index(joint_i=i + 1, side=side),
    #                 bodyosc2index(joint_i=i, side=side),
    #                 default_amplitude, phase_diff
    #             ])
    #             connectivity.append([
    #                 bodyosc2index(joint_i=i, side=side),
    #                 bodyosc2index(joint_i=i + 1, side=side),
    #                 default_amplitude, phase_diff
    #             ])
    #     # i+1 - i+1 (final)
    #     connectivity.append([
    #         bodyosc2index(joint_i=n_body_joints - 1, side=1),
    #         bodyosc2index(joint_i=n_body_joints - 1, side=0),
    #         default_amplitude, np.pi
    #     ])
    #     connectivity.append([
    #         bodyosc2index(joint_i=n_body_joints - 1, side=0),
    #         bodyosc2index(joint_i=n_body_joints - 1, side=1),
    #         default_amplitude, np.pi
    #     ])
    #
    #     # Legs (internal)
    #     for leg_i in range(2):
    #         for side_i in range(2):
    #             # 0 - 0
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
    #                 default_amplitude, np.pi
    #             ])
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
    #                 default_amplitude, np.pi
    #             ])
    #             # 0 - 1
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
    #                 default_amplitude, 0.5 * np.pi
    #             ])
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
    #                 default_amplitude, -0.5 * np.pi
    #             ])
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
    #                 default_amplitude, 0.5 * np.pi
    #             ])
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
    #                 default_amplitude, -0.5 * np.pi
    #             ])
    #             # 1 - 1
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
    #                 default_amplitude, np.pi
    #             ])
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
    #                 default_amplitude, np.pi
    #             ])
    #             # 1 - 2
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
    #                 default_amplitude, 0
    #             ])
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
    #                 default_amplitude, 0
    #             ])
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
    #                 default_amplitude, 0
    #             ])
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
    #                 default_amplitude, 0
    #             ])
    #             # 2 - 2
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
    #                 default_amplitude, np.pi
    #             ])
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
    #                 default_amplitude, np.pi
    #             ])
    #
    #     # Opposite leg interaction
    #     # TODO
    #
    #     # Following leg interaction
    #     # TODO
    #
    #     # Body-legs interaction
    #     for side_i in range(2):
    #         # Forelimbs
    #         connectivity.append([
    #             legosc2index(leg_i=0, side_i=side_i, joint_i=0, side=0),
    #             bodyosc2index(joint_i=1, side=side_i),
    #             default_amplitude, np.pi
    #         ])
    #         connectivity.append([
    #             bodyosc2index(joint_i=1, side=side_i),
    #             legosc2index(leg_i=0, side_i=side_i, joint_i=0, side=0),
    #             default_amplitude, np.pi
    #         ])
    #         connectivity.append([
    #             legosc2index(leg_i=0, side_i=side_i, joint_i=0, side=1),
    #             bodyosc2index(joint_i=1, side=side_i),
    #             default_amplitude, 0
    #         ])
    #         connectivity.append([
    #             bodyosc2index(joint_i=1, side=side_i),
    #             legosc2index(leg_i=0, side_i=side_i, joint_i=0, side=1),
    #             default_amplitude, 0
    #         ])
    #         # Hind limbs
    #         connectivity.append([
    #             legosc2index(leg_i=1, side_i=side_i, joint_i=0, side=0),
    #             bodyosc2index(joint_i=4, side=side_i),
    #             default_amplitude, np.pi
    #         ])
    #         connectivity.append([
    #             bodyosc2index(joint_i=4, side=side_i),
    #             legosc2index(leg_i=1, side_i=side_i, joint_i=0, side=0),
    #             default_amplitude, np.pi
    #         ])
    #         connectivity.append([
    #             legosc2index(leg_i=1, side_i=side_i, joint_i=0, side=1),
    #             bodyosc2index(joint_i=4, side=side_i),
    #             default_amplitude, 0
    #         ])
    #         connectivity.append([
    #             bodyosc2index(joint_i=4, side=side_i),
    #             legosc2index(leg_i=1, side_i=side_i, joint_i=0, side=1),
    #             default_amplitude, 0
    #         ])
    #     return connectivity
    #
    # @staticmethod
    # def swimming_parameters():
    #     """Swimming parameters"""
    #     raise Exception
    #     n_body_joints = 11
    #     connectivity = []
    #     default_amplitude = 3e2
    #
    #     # Body
    #     for i in range(n_body_joints - 1):
    #         # i - i
    #         connectivity.append([
    #             bodyosc2index(joint_i=i, side=1),
    #             bodyosc2index(joint_i=i, side=0),
    #             default_amplitude, np.pi
    #         ])
    #         connectivity.append([
    #             bodyosc2index(joint_i=i, side=0),
    #             bodyosc2index(joint_i=i, side=1),
    #             default_amplitude, np.pi
    #         ])
    #         # i - i+1
    #         for side in range(2):
    #             connectivity.append([
    #                 bodyosc2index(joint_i=i + 1, side=side),
    #                 bodyosc2index(joint_i=i, side=side),
    #                 default_amplitude, 2 * np.pi / n_body_joints
    #             ])
    #             connectivity.append([
    #                 bodyosc2index(joint_i=i, side=side),
    #                 bodyosc2index(joint_i=i + 1, side=side),
    #                 default_amplitude, -2 * np.pi / n_body_joints
    #             ])
    #     # i+1 - i+1 (final)
    #     connectivity.append([
    #         bodyosc2index(joint_i=n_body_joints - 1, side=1),
    #         bodyosc2index(joint_i=n_body_joints - 1, side=0),
    #         default_amplitude, np.pi
    #     ])
    #     connectivity.append([
    #         bodyosc2index(joint_i=n_body_joints - 1, side=0),
    #         bodyosc2index(joint_i=n_body_joints - 1, side=1),
    #         default_amplitude, np.pi
    #     ])
    #
    #     # Legs (internal)
    #     for leg_i in range(2):
    #         for side_i in range(2):
    #             # 0 - 0
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
    #                 default_amplitude, np.pi
    #             ])
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
    #                 default_amplitude, np.pi
    #             ])
    #             # 0 - 1
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
    #                 default_amplitude, 0.5 * np.pi
    #             ])
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=0),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
    #                 default_amplitude, -0.5 * np.pi
    #             ])
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
    #                 default_amplitude, 0.5 * np.pi
    #             ])
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=0, side=1),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
    #                 default_amplitude, -0.5 * np.pi
    #             ])
    #             # 1 - 1
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
    #                 default_amplitude, np.pi
    #             ])
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
    #                 default_amplitude, np.pi
    #             ])
    #             # 1 - 2
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
    #                 default_amplitude, 0
    #             ])
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=0),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
    #                 default_amplitude, 0
    #             ])
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
    #                 default_amplitude, 0
    #             ])
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=1, side=1),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
    #                 default_amplitude, 0
    #             ])
    #             # 2 - 2
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
    #                 default_amplitude, 0
    #             ])
    #             connectivity.append([
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=0),
    #                 legosc2index(leg_i=leg_i, side_i=side_i, joint_i=2, side=1),
    #                 default_amplitude, 0
    #             ])
    #
    #     # Opposite leg interaction
    #     # TODO
    #
    #     # Following leg interaction
    #     # TODO
    #
    #     # Body-legs interaction
    #     for side_i in range(2):
    #         # Forelimbs
    #         connectivity.append([
    #             legosc2index(leg_i=0, side_i=side_i, joint_i=0, side=0),
    #             bodyosc2index(joint_i=1, side=side_i),
    #             default_amplitude, np.pi
    #         ])
    #         connectivity.append([
    #             bodyosc2index(joint_i=1, side=side_i),
    #             legosc2index(leg_i=0, side_i=side_i, joint_i=0, side=0),
    #             default_amplitude, np.pi
    #         ])
    #         connectivity.append([
    #             legosc2index(leg_i=0, side_i=side_i, joint_i=0, side=1),
    #             bodyosc2index(joint_i=1, side=side_i),
    #             default_amplitude, 0
    #         ])
    #         connectivity.append([
    #             bodyosc2index(joint_i=1, side=side_i),
    #             legosc2index(leg_i=0, side_i=side_i, joint_i=0, side=1),
    #             default_amplitude, 0
    #         ])
    #         # Hind limbs
    #         connectivity.append([
    #             legosc2index(leg_i=1, side_i=side_i, joint_i=0, side=0),
    #             bodyosc2index(joint_i=4, side=side_i),
    #             default_amplitude, np.pi
    #         ])
    #         connectivity.append([
    #             bodyosc2index(joint_i=4, side=side_i),
    #             legosc2index(leg_i=1, side_i=side_i, joint_i=0, side=0),
    #             default_amplitude, np.pi
    #         ])
    #         connectivity.append([
    #             legosc2index(leg_i=1, side_i=side_i, joint_i=0, side=1),
    #             bodyosc2index(joint_i=4, side=side_i),
    #             default_amplitude, 0
    #         ])
    #         connectivity.append([
    #             bodyosc2index(joint_i=4, side=side_i),
    #             legosc2index(leg_i=1, side_i=side_i, joint_i=0, side=1),
    #             default_amplitude, 0
    #         ])
    #     return connectivity

    @staticmethod
    def load_params():
        opt = ModelOptions()
        # body_connections = _options['connec_body']
        # limb_connections = _options['connec_left_forelimb'] + _options['connec_right_forelimb'] + _options['connec_left_hindlimb'] + _options['connec_right_hindlimb']
        # body_to_limb_connections = _options['connec_body_left_forelimb'] + _options['connec_body_right_forelimb'] + _options['connec_body_left_hindlimb'] + _options['connec_body_right_hindlimb']
        # limb_to_limb_connections = _options['connec_inter_limb']
        # connectivity = body_connections + limb_connections + body_to_limb_connections + limb_to_limb_connections
        n_body = opt['n_body']
        n_dof_leg = opt['n_dof_legs']
        n_leg = 4
        w_body = opt['weigths_body']
        w_leg = opt['weigths_limb']
        phi_contra_body = opt['phi_contra_body']
        phi_up_body = opt['phi_up_body']
        phi_down_body = opt['phi_down_body']

        connectivity = np.zeros([1, 4])
        # body connectivity
        for i in np.arange(n_body):
            # contralateral connectivity
            connectivity = np.vstack((connectivity, [i, i + n_body, w_body, phi_contra_body]))
            connectivity = np.vstack((connectivity, [i + n_body, i, w_body, phi_contra_body]))
            if i < n_body - 1:
                # left side of body
                connectivity = np.vstack((connectivity, [i, i + 1, w_body, phi_down_body]))
                connectivity = np.vstack((connectivity, [i + 1, i, w_body, phi_up_body]))
                # right side of body
                connectivity = np.vstack((connectivity, [i + n_body, i + n_body + 1, w_body, phi_down_body]))
                connectivity = np.vstack((connectivity, [i + n_body + 1, i + n_body, w_body, phi_up_body]))

        leg_offset = 2 * n_body
        left_forelimb_offset = 0 + leg_offset
        right_forelimb_offset = (2 * n_dof_leg) + leg_offset
        left_hindlimb_offset = 2 * (2 * n_dof_leg) + leg_offset
        right_hindlimb_offset = 3 * (2 * n_dof_leg) + leg_offset

        for i in np.arange(2 * n_dof_leg):
            if i == 0 or np.mod(i, 3) == 0:  # shoulder
                # left forelimb shoulder
                connectivity = np.vstack(
                    (connectivity,
                     [i + left_forelimb_offset, i + left_forelimb_offset + 1, w_leg, opt['phi_shoulder_up']]))
                connectivity = np.vstack(
                    (connectivity,
                     [i + left_forelimb_offset + 1, i + left_forelimb_offset, w_leg, opt['phi_shoulder_down']]))
                # right forelimb shoulder
                connectivity = np.vstack(
                    (connectivity,
                     [i + right_forelimb_offset, i + right_forelimb_offset + 1, w_leg, opt['phi_shoulder_up']]))
                connectivity = np.vstack(
                    (connectivity, [i + right_forelimb_offset + 1, i + right_forelimb_offset, w_leg,
                                    opt['phi_shoulder_down']]))
                # left hindlim shoulder
                connectivity = np.vstack(
                    (connectivity,
                     [i + left_hindlimb_offset, i + left_hindlimb_offset + 1, w_leg, opt['phi_shoulder_up']]))
                connectivity = np.vstack(
                    (connectivity,
                     [i + left_hindlimb_offset + 1, i + left_hindlimb_offset, w_leg, opt['phi_shoulder_down']]))
                # right hindlimb shoulder
                connectivity = np.vstack(
                    (connectivity,
                     [i + right_hindlimb_offset, i + right_hindlimb_offset + 1, w_leg, opt['phi_shoulder_up']]))
                connectivity = np.vstack(
                    (connectivity, [i + right_hindlimb_offset + 1, i + right_hindlimb_offset, w_leg,
                                    opt['phi_shoulder_down']]))

            if i == 1 or i == 1 + n_dof_leg:  # knee
                connectivity = np.vstack(
                    (connectivity, [i + left_forelimb_offset, i + left_forelimb_offset + 1, w_leg, 0]))
                connectivity = np.vstack(
                    (connectivity, [i + left_forelimb_offset + 1, i + left_forelimb_offset, w_leg, 0]))
                # right forelimb shoulder
                connectivity = np.vstack(
                    (connectivity, [i + right_forelimb_offset, i + right_forelimb_offset + 1, w_leg, 0]))
                connectivity = np.vstack(
                    (connectivity, [i + right_forelimb_offset + 1, i + right_forelimb_offset, w_leg, 0]))
                # left hindlim shoulder
                connectivity = np.vstack(
                    (connectivity, [i + left_hindlimb_offset, i + left_hindlimb_offset + 1, w_leg, 0]))
                connectivity = np.vstack(
                    (connectivity, [i + left_hindlimb_offset + 1, i + left_hindlimb_offset, w_leg, 0]))
                # right hindlimb shoulder
                connectivity = np.vstack(
                    (connectivity, [i + right_hindlimb_offset, i + right_hindlimb_offset + 1, w_leg, 0]))
                connectivity = np.vstack(
                    (connectivity, [i + right_hindlimb_offset + 1, i + right_hindlimb_offset, w_leg, 0]))

        for i in np.arange(n_dof_leg):  # contralateral connexion
            if i == n_dof_leg - 1:
                phi = 0
            else:
                phi = np.pi
            # left forelimb
            connectivity = np.vstack(
                (connectivity, [i + left_forelimb_offset, i + left_forelimb_offset + n_dof_leg, w_leg, phi]))
            connectivity = np.vstack(
                (connectivity, [i + left_forelimb_offset + n_dof_leg, i + left_forelimb_offset, w_leg, phi]))

            connectivity = np.vstack(
                (connectivity, [i + right_forelimb_offset, i + right_forelimb_offset + n_dof_leg, w_leg, phi]))
            connectivity = np.vstack(
                (connectivity, [i + right_forelimb_offset + n_dof_leg, i + right_forelimb_offset, w_leg, phi]))

            connectivity = np.vstack(
                (connectivity, [i + left_hindlimb_offset, i + left_hindlimb_offset + n_dof_leg, w_leg, phi]))
            connectivity = np.vstack(
                (connectivity, [i + left_hindlimb_offset + n_dof_leg, i + left_hindlimb_offset, w_leg, phi]))

            connectivity = np.vstack(
                (connectivity, [i + right_hindlimb_offset, i + right_hindlimb_offset + n_dof_leg, w_leg, phi]))
            connectivity = np.vstack(
                (connectivity, [i + right_hindlimb_offset + n_dof_leg, i + right_hindlimb_offset, w_leg, phi]))

        # left forelimb connection to body
        connectivity = np.vstack((connectivity, [0, 22, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [0, 25, 3 * w_body, 0]))
        connectivity = np.vstack((connectivity, [1, 22, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [1, 25, 3 * w_body, 0]))
        # right forelimb connection to body
        connectivity = np.vstack((connectivity, [11, 28, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [11, 31, 3 * w_body, 0]))
        connectivity = np.vstack((connectivity, [12, 28, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [12, 31, 3 * w_body, 0]))
        # left hindlimb connections to body
        connectivity = np.vstack((connectivity, [6, 34, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [6, 37, 3 * w_body, 0]))
        connectivity = np.vstack((connectivity, [7, 34, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [7, 37, 3 * w_body, 0]))
        connectivity = np.vstack((connectivity, [8, 34, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [8, 37, 3 * w_body, 0]))
        connectivity = np.vstack((connectivity, [9, 34, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [9, 37, 3 * w_body, 0]))
        # right hindlimb connections to body
        connectivity = np.vstack((connectivity, [17, 40, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [17, 43, 3 * w_body, 0]))
        connectivity = np.vstack((connectivity, [18, 40, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [18, 43, 3 * w_body, 0]))
        connectivity = np.vstack((connectivity, [19, 40, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [19, 43, 3 * w_body, 0]))
        connectivity = np.vstack((connectivity, [20, 40, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [20, 43, 3 * w_body, 0]))

        # leg to leg connections
        connectivity = np.vstack((connectivity, [22, 34, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [34, 22, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [22, 28, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [28, 22, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [40, 34, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [34, 40, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [28, 40, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [40, 28, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [25, 37, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [37, 25, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [43, 37, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [37, 43, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [25, 31, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [31, 25, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [31, 43, 3 * w_body, np.pi]))
        connectivity = np.vstack((connectivity, [43, 31, 3 * w_body, np.pi]))

        debug = True
        if debug == True:
            print("-----connectivity-----")
            for i in np.arange(0,len(connectivity)):
                print(connectivity[i][:])
            print(np.shape(connectivity))
            #§  pdb.set_trace()

        return connectivity

    @classmethod
    def for_moving(cls):
        connectivity = cls.load_params()
        return cls(np.array(connectivity))

    @classmethod
    def for_walking(cls):
        """Parameters for walking"""
        connectivity = cls.walking_parameters()
        return cls(np.array(connectivity))

    @classmethod
    def for_swimming(cls):
        """Parameters for swimming"""
        connectivity = cls.swimming_parameters()
        return cls(np.array(connectivity))

    @property
    def connections(self):
        """Connections"""
        return self.array[:, [0, 1]]

    @property
    def weights(self):
        """Weights"""
        return self.array[:, 2]

    @property
    def desired_phases(self):
        """Weights"""
        return self.array[:, 3]


class JointsArray(NetworkArray):
    """Oscillator array"""

    @classmethod
    def from_parameters(cls, offsets, rates):
        """From each parameter"""
        return cls(np.array([offsets, rates]))

    @staticmethod
    def walking_parameters():
        """Walking parameters"""
        raise Exception
        n_body = 11
        n_dof_legs = 3
        n_legs = 4
        n_joints = n_body + n_legs * n_dof_legs
        options = SalamanderControlOptions.walking()
        offsets = np.zeros(n_joints)
        for leg_i in range(n_legs):
            for i in range(n_dof_legs):
                offsets[n_body + leg_i * n_dof_legs + i] = (
                    options["leg_{}_offset".format(i)]
                )
        rates = 10 * np.ones(n_joints)

        return offsets, rates

    @staticmethod
    def swimming_parameters():
        """Swimming parameters"""
        raise Exception
        n_body = 11
        n_dof_legs = 3
        n_legs = 4
        n_joints = n_body + n_legs * n_dof_legs
        options = SalamanderControlOptions.swimming()
        offsets = np.zeros(n_joints)
        for leg_i in range(n_legs):
            for i in range(n_dof_legs):
                offsets[n_body + leg_i * n_dof_legs + i] = (
                    options["leg_{}_offset".format(i)]
                )
        rates = 10 * np.ones(n_joints)
        return offsets, rates


    @staticmethod
    def load_params():
        _options = ModelOptions()
        offsets = _options['joints_offset']
        rates = _options['joints_rate']

        debug = True
        if debug == True:
            print("-----offsets-----")
            print(offsets)
            print('-----rates-----')
            print(rates)
            #pdb.set_trace()

        return offsets, rates

    @classmethod
    def for_moving(cls):
        offsets, rates = cls.load_params()
        return cls.from_parameters(offsets, rates)


    @classmethod
    def for_walking(cls):
        """Parameters for walking"""
        offsets, rates = cls.walking_parameters()
        return cls.from_parameters(offsets, rates)

    @classmethod
    def for_swimming(cls):
        """Parameters for swimming"""
        offsets, rates = cls.swimming_parameters()
        return cls.from_parameters(offsets, rates)

    @property
    def offsets(self):
        """Joints angles offsets"""
        return self.array[0]

    @property
    def rates(self):
        """Joints angles offsets rates"""
        return self.array[1]

    def set_body_offset(self, value, n_body_joints=11):
        """Body offset"""
        self.array[0, :n_body_joints] = value

    def update_drives(self, drive_speed, drive_turn):
        """
        :param drive_speed: drive that change the frequency
        :param drive_turn: drive that change the offset
        :return: send to the simulation the drive
        """
        drive_low_sat = 1
        drive_up_sat = 3
        # options = ModelOptions()
        # bodyosc2index, legosc2index
        if drive_low_sat <= drive_speed <= drive_up_sat:
            for leg_i in range(2):
                for side_i in range(2):
                    self.offsets[legjoint2index(leg_i, side_i, 0)] = 0
                    self.offsets[legjoint2index(leg_i, side_i, 1)] = np.pi/32
                    self.offsets[legjoint2index(leg_i, side_i, 2)] = np.pi/8
                    print('=====update walking offsets=====')
                    print(self.offsets)

        else:
            for leg_i in range(2):
                for side_i in range(2):
                    self.offsets[legjoint2index(leg_i, side_i, 0)] = -2*np.pi/5
                    self.offsets[legjoint2index(leg_i, side_i, 1)] = 0
                    self.offsets[legjoint2index(leg_i, side_i, 2)] = 0
                    print('=====update swimming offsets=====')
                    print(self.offsets)
                    #pdb.set_trace()


class SalamanderNetworkODE(ODESolver):
    """Salamander network"""

    def __init__(self, state, parameters, timestep):
        super(SalamanderNetworkODE, self).__init__(
            ode=ODE(rk4, ode_oscillators_sparse),
            state=state.array,
            timestep=timestep,
            parameters=parameters.to_ode_parameters()
        )
        self.state = state
        self.parameters = parameters
        self._n_oscillators = state.n_oscillators
        self._n_joints = parameters.joints.shape()[1]
        n_body = 11
        n_legs_dofs = 3
        # n_legs = 4
        self.group0 = [
            bodyosc2index(joint_i=i, side=0)
            for i in range(11)
        ] + [
            legosc2index(leg_i=leg_i, side_i=side_i, joint_i=joint_i, side=0)
            for leg_i in range(2)
            for side_i in range(2)
            for joint_i in range(n_legs_dofs)
        ]
        self.group1 = [
            bodyosc2index(joint_i=i, side=1)
            for i in range(n_body)
        ] + [
            legosc2index(leg_i=leg_i, side_i=side_i, joint_i=joint_i, side=1)
            for leg_i in range(2)
            for side_i in range(2)
            for joint_i in range(n_legs_dofs)
        ]

    @classmethod
    def from_gait(cls, gait, n_iterations, timestep):
        """ Salamander network from gait"""
        return (
            cls.swimming(n_iterations, timestep)
            if gait == "swimming"
            else cls.walking(n_iterations, timestep)
        )

    def update_gait(self, gait):
        """Update from gait"""
        self.parameters.update_gait(gait)
        self._parameters = self.parameters.to_ode_parameters()

    @classmethod
    def walking(cls, n_iterations, timestep):
        """Salamander swimming network"""
        state = OscillatorNetworkState.default_state(n_iterations)
        parameters = SalamanderNetworkParameters.for_walking()
        #TODO
        #parameters = 
        return cls(state, parameters, timestep)

    @classmethod
    def swimming(cls, n_iterations, timestep):
        """Salamander swimming network"""
        state = OscillatorNetworkState.default_state(n_iterations)
        parameters = SalamanderNetworkParameters.for_swimming()
        return cls(state, parameters, timestep)

    def control_step(self):
        """Control step"""
        self.step()
        return self.current_state

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
        return self._state[:, 0, self._n_oscillators:2 * self._n_oscillators]

    @property
    def damplitudes(self):
        """Amplitudes velocity"""
        return self._state[:, 1, self._n_oscillators:2 * self._n_oscillators]

    @property
    def offsets(self):
        """Offset"""
        return self._state[:, 0, 2 * self._n_oscillators:]

    @property
    def doffsets(self):
        """Offset velocity"""
        return self._state[:, 1, 2 * self._n_oscillators:]

    def get_outputs(self):
        """Outputs"""
        return self.amplitudes[self.iteration] * (
                1 + np.cos(self.phases[self.iteration])
        )

    def get_outputs_all(self):
        """Outputs"""
        return self.amplitudes * (
                1 + np.cos(self.phases)
        )

    def get_doutputs(self):
        """Outputs velocity"""
        return self.damplitudes[self.iteration] * (
                1 + np.cos(self.phases[self.iteration])
        ) - (
                       self.amplitudes[self.iteration]
                       * np.sin(self.phases[self.iteration])
                       * self.dphases[self.iteration]
               )

    def get_doutputs_all(self):
        """Outputs velocity"""
        return self.damplitudes * (
                1 + np.cos(self.phases)
        ) - self.amplitudes * np.sin(self.phases) * self.dphases

    def get_position_output(self):
        """Position output"""
        outputs = self.get_outputs()
        return (
                0.5 * (outputs[self.group0] - outputs[self.group1])
                + self.offsets[self.iteration]
        )

    def get_position_output_all(self):
        """Position output"""
        outputs = self.get_outputs_all()
        return (
                0.5 * (outputs[:, self.group0] - outputs[:, self.group1])
                + self.offsets
        )

    def get_velocity_output(self):
        """Position output"""
        outputs = self.get_doutputs()
        return 0.5 * (outputs[self.group0] - outputs[self.group1])

    def get_velocity_output_all(self):
        """Position output"""
        outputs = self.get_doutputs_all()
        return 0.5 * (outputs[:, self.group0] - outputs[:, self.group1])

    def update_drive(self, drive_speed, drive_turn):
        """Update drives"""
        self.parameters.oscillators.update_drives(drive_speed, drive_turn)
        self.parameters.joints.update_drives(drive_speed, drive_turn)
