"""Model options"""

import numpy as np


class ModelOptions(dict):
    """Simulation options"""
    """Parameters"""
    _getattr_ = dict.__getitem__
    _setattr_ = dict.__setitem__

    def __init__(self, **kwargs):
        super(ModelOptions, self).__init__()
        self["gait"] = kwargs.pop("gait", "walking")
        self["frequency"] = kwargs.pop("frequency", 1)
        self["body_amplitude_0"] = kwargs.pop("body_amplitude_0", 0)
        self["body_amplitude_1"] = kwargs.pop("body_amplitude_1", 0)
        self["body_stand_amplitude"] = kwargs.pop("body_stand_amplitude", 0.2)
        self["body_stand_shift"] = kwargs.pop("body_stand_shift", np.pi/4)

        # Legs
        self["leg_0_amplitude"] = kwargs.pop("leg_0_amplitude", 0.8)
        self["leg_0_offset"] = kwargs.pop("leg_0_offset", 0)

        self["leg_1_amplitude"] = kwargs.pop("leg_1_amplitude", np.pi/32)
        self["leg_1_offset"] = kwargs.pop("leg_1_offset", np.pi/32)

        self["leg_2_amplitude"] = kwargs.pop("leg_2_amplitude", np.pi/8)
        self["leg_2_offset"] = kwargs.pop("leg_2_offset", np.pi/8)

        #oscillators properties
        self["n_body"] = kwargs.pop("n_body", 11)
        self['n_dof_legs'] = kwargs.pop("n_dof_legs", 3)
        self['n_legs'] = kwargs.pop("n_legs", 4)
        #oscillators array
        #freq
        self['body_freqs'] = kwargs.pop('body_freqs', 2)
        self['limb_freqs'] = kwargs.pop('limb_freqs', 1)
        self['rates'] = kwargs.pop('rates', 10)
        #rates
        #amplitudes
        self['left_forelimb_amp'] = kwargs.pop('left_forelimb_amp', [self["leg_0_amplitude"], self["leg_1_offset"], self["leg_2_amplitude"],
                                                                     self["leg_0_amplitude"], self["leg_1_offset"], self["leg_2_amplitude"]])
        self['right_forelimb_amp'] = kwargs.pop('left_forelimb_amp', [self["leg_0_amplitude"], self["leg_1_offset"], self["leg_2_amplitude"],
                                                                     self["leg_0_amplitude"], self["leg_1_offset"], self["leg_2_amplitude"]])
        self['left_hindlimb_amp'] = kwargs.pop('left_forelimb_amp', [self["leg_0_amplitude"], self["leg_1_offset"], self["leg_2_amplitude"],
                                                                     self["leg_0_amplitude"], self["leg_1_offset"], self["leg_2_amplitude"]])
        self['right_hindlimb_amp'] = kwargs.pop('left_forelimb_amp', [self["leg_0_amplitude"], self["leg_1_offset"], self["leg_2_amplitude"],
                                                                     self["leg_0_amplitude"], self["leg_1_offset"], self["leg_2_amplitude"]])
        #self['forelimb_amp'] = np.append(self['left_forelimb_amp'], self['right_forelimb_amp'])                                          
        self['body_amp'] = kwargs.pop('body_amp', [np.abs(self["body_stand_amplitude"]*np.sin(2*np.pi*n/self["n_body"]-self["body_stand_shift"])) 
                                                  for n in np.arange(self["n_body"])])
        #self['amplitudes'] = np.append(self['left_forelimb_amp'], self['right_forelimb_amp'])
        #weigths for limb
        self['weigths_body'] = kwargs.pop('weigths_body', 300)
        self['weigths_limb'] = kwargs.pop('weigths_limb', 300)
        self['weights_limb_to_body'] = kwargs.pop('weights_limb_to_body', 600)
        self['weights_limb_to_limb'] = kwargs.pop('weights_limb_to_limb', 300)
        self['phi_contra_body'] = kwargs.pop('phi_contra_body', np.pi)
        self['phi_down_body'] = kwargs.pop('phi_down_body', -2*np.pi/self['n_body'])
        self['phi_up_body'] = kwargs.pop('phi_up_body', 2*np.pi/self['n_body'])
        #shoulder dephasing
        self['phi_shoulder_up'] = kwargs.pop('phi_shoulder_up', -np.pi/2)
        self['phi_shoulder_down'] = kwargs.pop('phi_shoulder_down', np.pi/2)
        self['phi_shoulder_contra'] = kwargs.pop('phi_shoulder_contra', np.pi)
        #phi limb to body
        self['phi_limb_to_body'] = kwargs.pop('phi_limb_to_body', np.pi)
        self['phi_limb_to_limb'] = kwargs.pop('phi_limb_to_limb', np.pi)
        #connectivity
        self['connec_body'] = kwargs.pop('connec_body', 
                                    [[11, 0, self['weigths_body'], self['phi_contra_body']], [0, 11, self['weigths_body'], self['phi_contra_body']],
                                     [1, 0, self['weigths_body'], self['phi_up_body']], [0, 1, self['weigths_body'], self['phi_down_body']],
                                     [12, 11, self['weigths_body'], self['phi_up_body']], [11, 12, self['weigths_body'], self['phi_down_body']],
                                     [12, 1, self['weigths_body'], self['phi_contra_body']], [1, 12, self['weigths_body'], self['phi_contra_body']],
                                     [2, 1, self['weigths_body'], np.pi], [1, 2, self['weigths_body'], np.pi], 
                                     [13, 12, self['weigths_body'], np.pi], [12, 13, self['weigths_body'], np.pi],
                                     [13, 2, self['weigths_body'], self['phi_contra_body']], [2, 13, self['weigths_body'], self['phi_contra_body']], 
                                     [3, 2, self['weigths_body'], self['phi_up_body']], [2, 3, self['weigths_body'], self['phi_down_body']],
                                     [14, 13, self['weigths_body'], self['phi_up_body']], [13, 14, self['weigths_body'], self['phi_down_body']],
                                     [14, 3, self['weigths_body'], self['phi_contra_body']], [3, 14, self['weigths_body'], self['phi_contra_body']],
                                     [4, 3, self['weigths_body'], self['phi_up_body']], [3, 4, self['weigths_body'], self['phi_down_body']],
                                     [15, 14, self['weigths_body'], self['phi_up_body']], [14, 15, self['weigths_body'], self['phi_down_body']],
                                     [15, 4, self['weigths_body'], self['phi_contra_body']], [4, 15, self['weigths_body'], self['phi_contra_body']],
                                     [5, 4, self['weigths_body'], self['phi_up_body']], [4, 5, self['weigths_body'], self['phi_down_body']],
                                     [16, 15, self['weigths_body'], self['phi_up_body']], [15, 16, self['weigths_body'], self['phi_down_body']],
                                     [16, 5, self['weigths_body'], self['phi_contra_body']], [5, 16, self['weigths_body'], self['phi_contra_body']],
                                     [6, 5, self['weigths_body'], self['phi_up_body']], [5, 6, self['weigths_body'], self['phi_down_body']],
                                     [17, 16, self['weigths_body'], self['phi_up_body']], [16, 17, self['weigths_body'], self['phi_down_body']],
                                     [17, 6, self['weigths_body'], self['phi_contra_body']], [6, 17, self['weigths_body'], self['phi_contra_body']],
                                     [7, 6, self['weigths_body'], self['phi_contra_body']], [6, 7, self['weigths_body'], self['phi_contra_body']],
                                     [18, 17, self['weigths_body'], self['phi_contra_body']], [17, 18, self['weigths_body'], self['phi_contra_body']],
                                     [18, 7, self['weigths_body'], self['phi_contra_body']], [7, 18, self['weigths_body'], self['phi_contra_body']],
                                     [8, 7, self['weigths_body'], self['phi_up_body']], [7, 8, self['weigths_body'], self['phi_down_body']],
                                     [19, 18, self['weigths_body'], self['phi_up_body']], [18, 19, self['weigths_body'], self['phi_down_body']],
                                     [19, 8, self['weigths_body'], self['phi_contra_body']], [8, 19, self['weigths_body'], self['phi_contra_body']],
                                     [9, 8, self['weigths_body'], self['phi_up_body']], [8, 9, self['weigths_body'], self['phi_down_body']],
                                     [20, 19, self['weigths_body'], self['phi_up_body']], [19, 20, self['weigths_body'], self['phi_down_body']],
                                     [20, 9, self['weigths_body'], self['phi_contra_body']], [9, 20, self['weigths_body'], self['phi_contra_body']],
                                     [10, 9, self['weigths_body'], self['phi_up_body']], [9, 10, self['weigths_body'], self['phi_down_body']],
                                     [21, 20, self['weigths_body'], self['phi_up_body']], [20, 21, self['weigths_body'], self['phi_down_body']],
                                     [21, 10, self['weigths_body'], self['phi_contra_body']], [10, 21, self['weigths_body'], self['phi_contra_body']]])
        #connection in the limbs itself
        self['connec_left_forelimb'] = kwargs.pop('connec_left_forelimb', [[22, 25, self['weigths_limb'], self['phi_shoulder_contra']],             
                                                                           [25, 22, self['weigths_limb'], self['phi_shoulder_contra']], 
                                                                           [22, 23, self['weigths_limb'], self['phi_shoulder_up']], 
                                                                           [23, 22, self['weigths_limb'], self['phi_shoulder_down']], 
                                                                           [25, 26, self['weigths_limb'], self['phi_shoulder_up']], 
                                                                           [26, 25, self['weigths_limb'], self['phi_shoulder_down']],
                                                                           [23, 26, self['weigths_limb'], self['phi_shoulder_contra']], 
                                                                           [26, 23, self['weigths_limb'], self['phi_shoulder_contra']], 
                                                                           [23, 24, self['weigths_limb'], 0], 
                                                                           [24, 23, self['weigths_limb'], 0], 
                                                                           [26, 27, self['weigths_limb'], 0], 
                                                                           [27, 26, self['weigths_limb'], 0],
                                                                           [24, 27, self['weigths_limb'], np.pi], 
                                                                           [27, 24, self['weigths_limb'], np.pi]])

        self['connec_right_forelimb'] = kwargs.pop('connec_right_forelimb', [[28, 31, self['weigths_limb'], self['phi_shoulder_contra']], 
                                                                             [31, 28, self['weigths_limb'], self['phi_shoulder_contra']],
                                                                             [28, 29, self['weigths_limb'], self['phi_shoulder_up']], 
                                                                             [29, 28, self['weigths_limb'], self['phi_shoulder_down']], 
                                                                             [31, 32, self['weigths_limb'], self['phi_shoulder_up']], 
                                                                             [32, 31, self['weigths_limb'], self['phi_shoulder_down']],
                                                                             [29, 32, self['weigths_limb'], self['phi_shoulder_contra']], 
                                                                             [32, 29, self['weigths_limb'], self['phi_shoulder_contra']], 
                                                                             [29, 30, self['weigths_limb'], 0], 
                                                                             [30, 29, self['weigths_limb'], 0], 
                                                                             [32, 33, self['weigths_limb'], 0], 
                                                                             [33, 32, self['weigths_limb'], 0],
                                                                             [30, 33, self['weigths_limb'], np.pi], 
                                                                             [33, 30, self['weigths_limb'], np.pi]])

        self['connec_left_hindlimb'] = kwargs.pop('connec_left_hindlimb', [[37, 34, self['weigths_limb'], self['phi_shoulder_contra']], 
                                                                           [34, 37, self['weigths_limb'], self['phi_shoulder_contra']], 
                                                                           [35, 34, self['weigths_limb'], self['phi_shoulder_up']], 
                                                                           [34, 35, self['weigths_limb'], self['phi_shoulder_down']], 
                                                                           [38, 37, self['weigths_limb'], self['phi_shoulder_down']], 
                                                                           [37, 38, self['weigths_limb'], self['phi_shoulder_up']],
                                                                           [38, 35, self['weigths_limb'], self['phi_shoulder_contra']], 
                                                                           [35, 38, self['weigths_limb'], self['phi_shoulder_contra']], 
                                                                           [36, 35, self['weigths_limb'], 0], 
                                                                           [35, 36, self['weigths_limb'], 0], 
                                                                           [39, 38, self['weigths_limb'], 0], 
                                                                           [38, 39, self['weigths_limb'], 0],
                                                                           [39, 36, self['weigths_limb'], np.pi], 
                                                                           [36, 39, self['weigths_limb'], np.pi]])

        self['connec_right_hindlimb'] = kwargs.pop('connec_right_hindlimb', [[43, 40, self['weigths_limb'], self['phi_shoulder_contra']], 
                                                                             [40, 43, self['weigths_limb'], self['phi_shoulder_contra']], 
                                                                             [41, 40, self['weigths_limb'], self['phi_shoulder_down']], 
                                                                             [40, 41, self['weigths_limb'], self['phi_shoulder_up']],
                                                                             [44, 43, self['weigths_limb'], self['phi_shoulder_down']], 
                                                                             [43, 44, self['weigths_limb'], self['phi_shoulder_up']],
                                                                             [44, 41, self['weigths_limb'], self['phi_shoulder_contra']], 
                                                                             [41, 44, self['weigths_limb'], self['phi_shoulder_contra']], 
                                                                             [42, 41, self['weigths_limb'], 0], 
                                                                             [41, 42, self['weigths_limb'], 0], 
                                                                             [45, 44, self['weigths_limb'], 0], 
                                                                             [44, 45, self['weigths_limb'], 0],
                                                                             [45, 42, self['weigths_limb'], np.pi], 
                                                                             [42, 45, self['weigths_limb'], np.pi]])
        #connection from limb to body 
        self['connec_body_left_forelimb'] = kwargs.pop('connec_body_left_forelimb', [[22, 1, self['weights_limb_to_body'], self['phi_limb_to_body']]]) #, 
                                                                                     #[25, 1, self['weights_limb_to_body'], 0]])

        self['connec_body_right_forelimb'] = kwargs.pop('connec_body_right_forelimb',[[28, 12, self['weights_limb_to_body'], self['phi_limb_to_body']]])#,
                                                                                     # [31, 12, self['weights_limb_to_body'], 0]])

        self['connec_body_left_hindlimb'] = kwargs.pop('connec_body_left_hindlimb',[[34, 4, self['weights_limb_to_body'], self['phi_limb_to_body']]])#,
                                                                                   # [37, 4, self['weights_limb_to_body'], 0]])

        self['connec_body_right_hindlimb'] = kwargs.pop('connec_body_right_hindlimb',[[40, 15, self['weights_limb_to_body'], self['phi_limb_to_body']]])#, 
                                                                                      #[43, 15, self['weights_limb_to_body'], 0]])
        #to be check
        self['connec_inter_limb'] = kwargs.pop('connec_inter_limb', [[25, 34, self['weights_limb_to_limb'], self['phi_limb_to_limb']], 
                                                                     [34, 25, self['weights_limb_to_limb'], self['phi_limb_to_limb']], 
                                                                     [22, 37, self['weights_limb_to_limb'], self['phi_limb_to_limb']], 
                                                                     [37, 22, self['weights_limb_to_limb'], self['phi_limb_to_limb']],
                                                                     [31, 40, self['weights_limb_to_limb'], self['phi_limb_to_limb']], 
                                                                     [40, 31, self['weights_limb_to_limb'], self['phi_limb_to_limb']], 
                                                                     [28, 43, self['weights_limb_to_limb'], self['phi_limb_to_limb']], 
                                                                     [43, 28, self['weights_limb_to_limb'], self['phi_limb_to_limb']],
                                                                     [22, 28, self['weights_limb_to_limb'], self['phi_limb_to_limb']], 
                                                                     [28, 22, self['weights_limb_to_limb'], self['phi_limb_to_limb']], 
                                                                     [25, 31, self['weights_limb_to_limb'], self['phi_limb_to_limb']], 
                                                                     [31, 25, self['weights_limb_to_limb'], self['phi_limb_to_limb']],
                                                                     [34, 40, self['weights_limb_to_limb'], self['phi_limb_to_limb']], 
                                                                     [40, 34, self['weights_limb_to_limb'], self['phi_limb_to_limb']], 
                                                                     [37, 43, self['weights_limb_to_limb'], self['phi_limb_to_limb']], 
                                                                     [43, 37, self['weights_limb_to_limb'], self['phi_limb_to_limb']]])

        self['joints_body_offset'] = kwargs.pop('joints_body_offset', np.zeros(self['n_body']))
        self['joints_limb_offset'] = kwargs.pop('joints_limb_offset', [self["leg_{}_offset".format(i)] for i in range(self['n_dof_legs'])]*4)
        
        self['joints_offset'] = kwargs.pop('joints_offset',np.append(self['joints_body_offset'], self['joints_limb_offset']))
        self['joints_rate'] = kwargs.pop('joints_rate', self['rates']*np.ones(np.shape(self['joints_offset'])[0]))

    @property
    def frequency(self):
        """Model frequency"""
        return self["frequency"]

    @property
    def body_stand_amplitude(self):
        """Model body amplitude"""
        return self["body_stand_amplitude"]

    @property
    def gait(self):
        """Model gait"""
        return self["gait"]

    @gait.setter
    def gait(self, value):
        self["gait"] = value
