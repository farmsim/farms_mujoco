"""Simulator"""

import time

import numpy as np

import pybullet
import pybullet_data

import farms_pylog as pylog


def init_engine(headless=False, opengl2=False):
    """Initialise engine"""
    pylog.debug('Pybullet version: {}'.format(pybullet.getAPIVersion()))
    background_color = 0.9*np.ones(3)
    pybullet.connect(
        pybullet.DIRECT if headless else pybullet.GUI,  # pybullet.DIRECT
        # options='--enable_experimental_opencl'
        # options='--opengl2'  #  --minGraphicsUpdateTimeMs=32000
        options=(
            '--background_color_red={}'
            ' --background_color_green={}'
            ' --background_color_blue={}'
        ).format(*background_color) + (
            ' --opengl2' if opengl2 else ''
        )
    )
    pybullet_path = pybullet_data.getDataPath()
    pylog.debug('Adding pybullet data path {}'.format(pybullet_path))
    pybullet.setAdditionalSearchPath(pybullet_path)


def real_time_handing(timestep, tic_rt, rtl=1.0, verbose=False, **kwargs):
    """Real-time handling"""
    tic_rt[1] = time.time()
    tic_rt[2] += timestep/rtl - (tic_rt[1] - tic_rt[0])
    rtf = timestep / (tic_rt[1] - tic_rt[0])
    if tic_rt[2] > 1e-2:
        time.sleep(tic_rt[2])
        tic_rt[2] = 0
    elif tic_rt[2] < 0:
        tic_rt[2] = 0
    tic_rt[0] = time.time()
    if rtf < 0.1 and verbose:
        pylog.debug('Significantly slower than real-time: {} %'.format(100*rtf))
        time_plugin = kwargs.pop('time_plugin', False)
        time_control = kwargs.pop('time_control', False)
        time_sim = kwargs.pop('time_sim', False)
        if time_plugin:
            pylog.debug('  Time in py_plugins: {} [ms]'.format(time_plugin))
        if time_control:
            pylog.debug('    Time in control: {} [ms]'.format(time_control))
        if time_sim:
            pylog.debug('  Time in simulation: {} [ms]'.format(time_sim))
