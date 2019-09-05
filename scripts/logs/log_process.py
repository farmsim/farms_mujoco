import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
from scipy.signal import butter, lfilter, find_peaks
from farms_bullet.experiments.salamander.convention import leglink2index, leglink2name, legjoint2index


class Logs(dict):
    """Simulation options"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, path):
        super(Logs, self).__init__()
        self.path = path
        self.contacts = np.load('{}/contacts.npy'.format(path))
        self.joints = np.load('{}/joints.npy'.format(path))
        self.links = np.load('{}/links.npy'.format(path))
        self.times = np.load('{}/times.npy'.format(path))
        self.perf = {}
        self.strides = {}
        self.TO = {}
        self.HC = {}


class strides(dict):
    """strides"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self):
        super(strides, self).__init__()
        self.strides = {}


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff=20, fs=1000, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def plot_forces(logs):
    for leg_i in range(4):
        plt.figure()
        for force_i in [8]:
            plt.plot(logs.times, logs.contacts[:, leg_i, force_i], color='steelblue', label=force_i)
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='k', linestyle='--', linewidth=0.5)
            plt.minorticks_on()
            plt.legend()
        plt.show(block=False)


def events(contact, logs, split=500, tresh_factor = 0.1):
    peak_dict = {}
    for i in range(4):
        contact['C{}hc'.format(i)] = [0]
        contact['C{}to'.format(i)] = [0]
    for i in range(4):
        treshold = tresh_factor * np.max(contact['C{}'.format(i)][:])
        peaks, _ = find_peaks(contact['C{}'.format(i)], height=8 * treshold)
        peak_dict['leg_{}'.format(i)] = peaks
        for k in range(len(contact['C{}'.format(i)]) - 1):
            # computing heel contact HC
            if contact['C{}'.format(i)][k] < treshold and contact['C{}'.format(i)][k + 1] > treshold:
                old_hc = contact['C{}hc'.format(i)][-1]
                new_hc = k + 1
                if new_hc > old_hc + split:
                    contact['C{}hc'.format(i)].append(k + 1)

            # computing toe off TO
            if contact['C{}'.format(i)][k] > treshold and contact['C{}'.format(i)][k + 1] < treshold:
                old_to = contact['C{}to'.format(i)][-1]
                new_to = k + 1
                if new_to > old_to + split:  # and new_to > old_hc + 200:
                    contact['C{}to'.format(i)].append(k + 1)
        contact['C{}hc'.format(i)].pop(0)
        contact['C{}to'.format(i)].pop(0)
        if contact['C{}to'.format(i)][0] < contact['C{}hc'.format(i)][0]:
            contact['C{}to'.format(i)].pop(0)

        if contact['C{}hc'.format(i)][1] < contact['C{}to'.format(i)][0]:
            contact['C{}hc'.format(i)].pop(0)

        # adding the key value to the dictionnary
    for leg_i in range(4):
        logs.HC['leg_{}'.format(leg_i)] = contact['C{}hc'.format(leg_i)]
        logs.TO['leg_{}'.format(leg_i)] = contact['C{}to'.format(leg_i)]
    return contact, peak_dict


def duty_f(contact, logs, Verbose=False):
    """duty factor = stance duration/stride duration"""
    duty_f = {}
    stride_duration = {}
    mean_duty_f = {}
    mean_stride_duration = {}
    for leg_i in range(4):
        stride_duration['leg_{}'.format(leg_i)] = []
        duty_f['leg_{}'.format(leg_i)] = []
        stride_dur = np.diff(contact['C{}hc'.format(leg_i)])
        if len(contact['C{}hc'.format(leg_i)]) > len(contact['C{}to'.format(leg_i)]):
            for k in range(len(contact['C{}to'.format(leg_i)]) - 1):
                duty_computed = (contact['C{}to'.format(leg_i)][k] - contact['C{}hc'.format(leg_i)][k]) / \
                                stride_dur[k]
                duty_f['leg_{}'.format(leg_i)].append(duty_computed)
        else:
            for k in range(len(contact['C{}hc'.format(leg_i)]) - 1):
                duty_computed = (contact['C{}to'.format(leg_i)][k] - contact['C{}hc'.format(leg_i)][k]) / \
                                stride_dur[k]
                duty_f['leg_{}'.format(leg_i)].append(duty_computed)
        stride_duration['leg_{}'.format(leg_i)] = list(stride_dur)
        mean_duty_f['leg_{}'.format(leg_i)] = np.mean(duty_computed)
        mean_stride_duration['leg_{}'.format(leg_i)] = np.mean(stride_dur)

    logs.perf['mean_stride_duration'] = mean_stride_duration
    logs.perf['mean_duty_f'] = mean_duty_f
    logs.perf['stride_duration'] = stride_duration
    logs.perf['duty_f'] = duty_f

    return duty_f


def plot_duty(contact):
    plt.figure(figsize=[16, 9])
    leg_disp = [0, 2, 1, 3]
    y_label = ['LH', 'LF', 'RF', 'RH']
    for i in range(4):
        leg_i = leg_disp[i]
        #print('------leg{}------'.format(leg_i))
        ax = plt.subplot(4, 1, i + 1)
        if len(contact['C{}to'.format(leg_i)]) < len(contact['C{}hc'.format(leg_i)]):
            len_i = len(contact['C{}to'.format(leg_i)])
        else:
            len_i = len(contact['C{}hc'.format(leg_i)])

        for k in range(len_i):
            plt.fill_between([contact['C{}hc'.format(leg_i)][k],
                              contact['C{}hc'.format(leg_i)][k],
                              contact['C{}to'.format(leg_i)][k],
                              contact['C{}to'.format(leg_i)][k]],
                             [0, 1, 1, 0], color='steelblue')
            plt.ylim([0, 1])
        ax.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        plt.ylabel(y_label[i])
        ax.set_yticks([])
        if leg_i < 3:
            ax.set_xticks([])
        if leg_i == 3:
            plt.xlabel('Time [ms]')
    plt.axis('tight')
    plt.savefig(logs.path + '/duty.pdf', bbox_inches='tight')
    plt.show(block=False)


def plot_body_joints(logs):
    plt.figure()
    for i in range(11):
        plt.subplot(11, 1, i + 1)
        plt.plot(logs.joints[:, i, 0])  # , logs.joints[:,i,1])
        plt.axis('off')
    plt.savefig('bodyJoints.pdf', bbox_inches='tight')
    plt.show(block=False)


def plot_leg_joints(logs):
    plt.figure()
    for i in np.arange(11, 19):
        plt.subplot(11, 1, i - 10)
        plt.plot(logs.joints[:, i, 0])  # , logs.joints[:,i,1])
        plt.axis('off')
    plt.savefig('legJoints.pdf', bbox_inches='tight')
    plt.show(block=False)


def contact(logs):
    contact = {}
    for leg_i in range(4):
        contact['C{}'.format(leg_i)] = butter_lowpass_filter(np.abs(logs.contacts[:, leg_i, 8]), 5, 1000, 4)
    return contact


def xy_links(logs, link_i=0):
    plt.figure()
    plt.plot(logs.links[:, link_i, 0], logs.links[:, link_i, 1], label='link_{}'.format(link_i))
    plt.legend()
    plt.xlabel('x position [m]')
    plt.ylabel('y position [m]')
    plt.show(block=False)


def position_links(logs, link_i=0):
    # plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(logs.links[:, link_i, 0], logs.links[:, link_i, 1])
    # plt.plot(logs.links[:, link_i, 1])
    plt.subplot(2, 1, 2)
    plt.plot(logs.links[:, link_i, 2])
    plt.show(block=False)


def euler_links(logs, link_i=0):
    euler = []
    for i in range(np.shape(logs.links)[0]):
        euler.append(np.asarray(p.getEulerFromQuaternion(logs.links[i, link_i, 3:7])))
    return euler


def plot_euler(euler):
    plt.figure()
    plt.plot(euler[:][0])
    plt.plot(euler[:][1])
    plt.plot(euler[:][2])
    plt.show(block=False)


def vel_links(logs, link_i=0):
    velx_link = {}
    vely_link = {}
    velz_link = {}
    plt.figure()
    vel_id = ['x', 'y', 'z']
    for vel_i in np.arange(6, 9):
        plt.plot(logs.links[:, link_i, vel_i], label='velocity ' + vel_id[vel_i - 6])
    plt.legend()
    plt.show(block=False)
    velx_link['link_{}'.format(link_i)] = np.mean(logs.links[:, link_i, 6])
    vely_link['link_{}'.format(link_i)] = np.mean(logs.links[:, link_i, 7])
    velz_link['link_{}'.format(link_i)] = np.mean(logs.links[:, link_i, 8])

    logs.perf['avg_velx'] = velx_link
    logs.perf['avg_vely'] = vely_link
    logs.perf['avg_velz'] = velz_link

    # logs.perf['std_velx'] = np.std(logs.links[:, link_i, 6])
    # logs.perf['std_vely'] = np.std(logs.links[:, link_i, 7])
    # logs.perf['std_velz'] = np.std(logs.links[:, link_i, 8])


def show_perf(logs):
    for key in logs.perf.keys():
        #print('======{}======'.format(key))
        if logs.perf[key].keys() is not None:
            for s_keys in logs.perf[key].keys():
                print('{} = {}'.format(s_keys, logs.perf[key][s_keys]))


def invertTransform(logs):
    # p.invertTransform(logs.links[])
    return


plt.close('all')
logs = Logs(path='exp_d2,5_angle10')
contact = contact(logs)
contact, peaks_dict = events(contact, logs, split=450, tresh_factor=0.08)
contact_id = 2
duty_f = duty_f(contact, logs)

plot_duty(contact)

vel_links(logs, link_i=11)
start = 10
offset = 2

plt.figure(figsize=[12, 8])
for link_i in [15]:
    plt.subplot(2, 1, 1)
    plt.plot(logs.links[:, link_i, 0], logs.links[:, link_i, 1], 'k.', markersize=0.25)
    for hc in range(len(logs.HC['leg_0'])):
        plt.plot(logs.links[logs.HC['leg_0'][hc], link_i, 0], logs.links[logs.HC['leg_0'][hc], link_i, 1], 'r*')
    plt.xlabel('dimension x [m]')
    plt.ylabel('dimension y [m]')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.minorticks_on()

    plt.subplot(2, 1, 2)
    plt.plot(logs.links[:, link_i, 2])
    for hc in range(len(logs.HC['leg_0'])):
        plt.plot(logs.HC['leg_0'][hc], logs.links[logs.HC['leg_0'][hc], link_i, 2], 'b*')
    plt.xlabel('time [ms]')
    plt.ylabel('dimension z [m]')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.minorticks_on()
    plt.legend()
plt.savefig(logs.path + '/trajectories.pdf', bbox_inches='tight')
plt.show(block=False)

strides_length = {}
leg_id = [15, 19, 23, 27]
for leg_i in range(4):
    strides_length['leg_{}'.format(leg_i)] = []
    delta_x = np.diff(logs.links[logs.HC['leg_{}'.format(leg_i)], leg_id[leg_i], 0])
    delta_y = np.diff(logs.links[logs.HC['leg_{}'.format(leg_i)], leg_id[leg_i], 1])
    for hc in range(len(delta_x)):
        strides_length_i = np.sqrt(delta_x[hc] ** 2 + delta_y[hc] ** 2)
        strides_length['leg_{}'.format(leg_i)].append(strides_length_i)

logs.perf['strides_length'] = strides_length

plt.figure(figsize=[12, 8])
for contact_id in range(4):
    ax = plt.subplot(4, 1, contact_id + 1)
    plt.fill_between(range(len(contact['C{}'.format(contact_id)])), contact['C{}'.format(contact_id)],
                     color='steelblue')
    plt.plot(contact['C{}'.format(contact_id)])
    plt.plot(contact['C{}hc'.format(contact_id)], 2 * np.ones(len(contact['C{}hc'.format(contact_id)])), 'r*',
             label='Heel contact')
    plt.plot(contact['C{}to'.format(contact_id)], 2 * np.ones(len(contact['C{}to'.format(contact_id)])), 'bo',
             label='Toe off')
    if contact_id < 3:
        ax.set_xticks([])
    else:
        plt.xlabel('Time [ms]')
    plt.ylabel('Fc_{} [N]'.format(contact_id))
plt.savefig(logs.path+'/contacts.pdf', bbox_inches='tight')
plt.show(block=False)

# for leg_i in range(2):
#     for side_i in range(2):
#         print(legjoint2index(leg_i=leg_i, joint_i=0, side_i=side_i))


plt.figure(figsize=[16,9])
for i in range(4):
    plt.plot(logs.perf['strides_length']['leg_{}'.format(i)])
plt.show(block=False)

#printing everything for the report
for i in range(4):
    print('duty factor leg_{}:'.format(i),np.mean(logs.perf['duty_f']['leg_{}'.format(i)]))
for i in range(4):
    print('stride duration leg_{}:'.format(i),np.mean(logs.perf['stride_duration']['leg_{}'.format(i)]))
for i in range(4):
    print('stride length leg_{}:'.format(i),np.mean(logs.perf['strides_length']['leg_{}'.format(i)]))
    if i == 3:
        print(logs.perf['avg_velx'])
        print(logs.perf['avg_vely'])
        print(logs.perf['avg_velz'])
