"""Farms - Test salamander network"""

import time
import numpy as np
import matplotlib.pyplot as plt
from farms_bullet.controllers.network import SalamanderNetwork


def main():
    """Main"""
    timestep = 1e-3
    times = np.arange(0, 10, timestep)
    n_iterations = len(times)
    network = SalamanderNetwork.from_gait(gait="walking", timestep=1e-3)
    freqs = np.ones(np.shape(network.phases))
    tic = time.time()
    phase_log = np.zeros([n_iterations, len(freqs)])
    for i in range(n_iterations):
        phase_log[i, :] = network.control_step(freqs)
    print("Integration time: {} [s]".format(time.time() - tic))
    for i, data in enumerate(phase_log.T):
        plt.plot(times, data, label=r"$\theta{}$".format(i))
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Phase [rad]")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
