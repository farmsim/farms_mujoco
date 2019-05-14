#!/usr/bin/env python3
"""Run Simon's experiment in Bullet"""

from farms_bullet.simulations.simon import run_simon
import matplotlib.pyplot as plt


def main():
    """Main"""
    run_simon()
    plt.show()


if __name__ == '__main__':
    main()
