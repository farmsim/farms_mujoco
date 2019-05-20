#!/usr/bin/env python3
"""Run Simon's experiment in Bullet"""

import matplotlib.pyplot as plt
from farms_bullet.experiments.simon.simulation import run_simon


def main():
    """Main"""
    run_simon()
    plt.show()


if __name__ == '__main__':
    main()
