import time
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
from farms_bullet.controllers.network import SalamanderNetworkODE
from farms_bullet.animats.model_options2 import ModelOptions

def plot_data(times, data, body_ids, figurename, label, ylabel):
    """Plot data"""
    for i, _data in enumerate(data.T):
        if i < body_ids:
            plt.figure("{}-body".format(figurename))
        else:
            plt.figure("{}-legs".format(figurename))
        plt.plot(times, _data, label=r"{}{}".format(label, i))
    for figure in ["{}-body", "{}-legs"]:
        plt.figure(figure.format(figurename))
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel(ylabel)
        plt.legend()


def drive(start_drive, end_drive, timestep):
	t_drive = np.arange(start_drive, end_drive, timestep)
	drive = np.array([])
	for i in t_drive:
		drive = np.append(drive, 0.8 * i - 7/5)
	return drive


plt.close('all')
options = ModelOptions()
freqs = np.hstack((options['body_freqs']*np.ones(22),options['limb_freqs']*np.ones(24)))
timestep = 1e-3
time_end = 15
times = np.arange(0, time_end, timestep)
network = SalamanderNetworkODE.walking(
        n_iterations=len(times),
        timestep=timestep)
n_iterations = len(times)


# Simulate (method 1)
time_control = 0
start_drive = 3
end_drive = 8
drive = np.hstack((np.ones(int(start_drive/timestep)), drive(start_drive, end_drive, timestep), np.ones(int((time_end-end_drive)/timestep))))
for i in range(n_iterations-1):
    if i > 3000:
        network.update_drive(drive_speed= drive[i], drive_turn=0)
    
    network.parameters.oscillators.freqs = freqs
    network.control_step()

phases = network.phases
dphases = network.dphases
amplitudes = network.amplitudes
damplitudes = network.damplitudes
offsets = network.offsets
doffsets = network.doffsets
positions = network.get_position_output_all()
velocities = network.get_velocity_output_all()
outputs = network.get_outputs_all()
axis_verbose = 'off'
nbr_subfig = 20
data_shape = np.shape(amplitudes)
plt.figure(frameon=True)
for i in np.arange(11):
	plt.subplot(nbr_subfig,1,i+1)
	plt.plot(outputs[:,i])
	plt.axis(axis_verbose)

plt.subplot(nbr_subfig, 1, 12)
plt.plot([0, time_end],[0, 0],'k--')
plt.axis('off')
for i in np.arange(13,nbr_subfig-2):
	plt.subplot(nbr_subfig,1,i)
	plt.plot(outputs[:,i])
	plt.axis(axis_verbose)

plt.subplot(nbr_subfig, 1, 19)
plt.plot([0, time_end],[0, 0],'k--')
plt.axis('off')
plt.subplot(nbr_subfig,1,20)
plt.plot(drive)
plt.axis(axis_verbose)
plt.show(block=False)