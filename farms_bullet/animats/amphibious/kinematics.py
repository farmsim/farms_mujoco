"""Kinematics"""

import numpy as np
from scipy.signal import resample


class AmphibiousKinematics:
    """Amphibious kinematics"""

    def __init__(self, animat_options, animat_data, timestep):
        super(AmphibiousKinematics, self).__init__()
        self.kinematics = np.loadtxt(animat_options.control.kinematics_file)
        self.kinematics = self.kinematics[:, 3:]
        self.kinematics = ((self.kinematics + np.pi) % (2*np.pi)) - np.pi
        n_samples = 10*np.shape(self.kinematics)[0]
        self.kinematics = resample(self.kinematics, n_samples)
        self.animat_options = animat_options
        self.animat_data = animat_data
        self._timestep = timestep
        self._time = 0

    def control_step(self):
        """Control step"""
        self._time += self._timestep
        self.animat_data.iteration += 1
        if self.animat_data.iteration + 1 > np.shape(self.kinematics)[0]:
            self.animat_data.iteration = 0

    def get_outputs(self):
        """Outputs"""
        return self.kinematics[self.animat_data.iteration]

    def get_outputs_all(self):
        """Outputs"""
        return self.kinematics[:] % np.pi

    def get_doutputs(self):
        """Outputs velocity"""
        return (
            (
                self.kinematics[self.animat_data.iteration]
                - self.kinematics[self.animat_data.iteration-1]
            )/self._timestep
            if self.animat_data.iteration
            else np.zeros_like(self.kinematics[0])
        )

    def get_doutputs_all(self):
        """Outputs velocity"""
        return np.diff(self.kinematics)

    def get_position_output(self):
        """Position output"""
        return self.get_outputs()

    def get_position_output_all(self):
        """Position output"""
        return self.get_outputs_all()

    def get_velocity_output(self):
        """Position output"""
        return self.get_doutputs()

    def get_velocity_output_all(self):
        """Position output"""
        return self.get_doutputs_all()

    def update(self, options):
        """Update drives"""
