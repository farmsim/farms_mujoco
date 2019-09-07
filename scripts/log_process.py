import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp


class LogsProcess:
    def __init__(self):
        super(self, LogsProcess).__init__()

    @staticmethod
    def import_contacts(path):
        return np.load('{}/contacts.npy'.format(path))

    @staticmethod
    def import_joints(path):
        return np.load('{}/joints.npy'.format(path))

    @staticmethod
    def import_links(path):
        return np.load('{}/links.npy'.format(path))

    @staticmethod
    def import_times(path):
        return np.load('{}/times.npy'.format(path))

    @staticmethod
    def plot_forces(contacts, HeelCont=None, ToeOff=None):
        for leg_i in range(4):
            plt.figure()
            for force_i in [8]:
                plt.plot(times, contacts[:, leg_i, force_i], color='steelblue', label=force_i)

                # if HeelCont is not None:
                #     plt.plot(HeelCont['leg_{}'.format(leg_i + 1)],
                #              contacts[HeelCont['leg_{}'.format(leg_i + 1)], leg_i, force_i],
                #              'bo', linewidth=2)
                #
                # if ToeOff is not None:
                #     plt.plot(ToeOff['leg_{}'.format(leg_i + 1)],
                #              contacts[ToeOff['leg_{}'.format(leg_i + 1)], leg_i, force_i],
                #              'r*', linewidth=2)

                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='k', linestyle='--', linewidth=0.5)
                plt.minorticks_on()
                plt.legend()
            plt.show(block=False)

    @staticmethod
    def stance_duration(contacts):
        Fsum_z = 8
        HeelCont = {'leg_1': [], 'leg_2': [], 'leg_3': [], 'leg_4': []}
        ToeOff = {'leg_1': [], 'leg_2': [], 'leg_3': [], 'leg_4': []}
        for leg_i in range(4):
            treshold = 0.2 * np.max(contacts[:, leg_i, Fsum_z])
            for dt in range(np.shape(contacts)[0] - 1):
                if contacts[dt, leg_i, Fsum_z] < treshold and contacts[dt + 1, leg_i, Fsum_z] > treshold:
                    HeelCont['leg_{}'.format(leg_i + 1)].append(dt + 1)
                if contacts[dt, leg_i, Fsum_z] > treshold and contacts[dt + 1, leg_i, Fsum_z] < treshold:
                    ToeOff['leg_{}'.format(leg_i + 1)].append(dt + 1)
        return HeelCont, ToeOff


plt.close('all')
path = 'logs/exp01/'
contacts = LogsProcess.import_contacts(path)
joints = LogsProcess.import_joints(path)
links = LogsProcess.import_links(path)
times = LogsProcess.import_times(path)

HeelCont, ToeOff = LogsProcess.stance_duration(contacts)
LogsProcess.plot_forces(contacts, HeelCont, ToeOff)

# plt.figure()
# for joints in range(11):
#     plt.plot(joints[:, link_body_i, 0])
#
# plt.show(block=False)
