"""Simulator"""

import time

import numpy as np

import pybullet
import pybullet_data


def init_engine(headless=False):
    """Initialise engine"""
    print(pybullet.getAPIVersion())
    background_color = 0.9*np.ones(3)
    pybullet.connect(
        pybullet.DIRECT if headless else pybullet.GUI,
        # options="--enable_experimental_opencl"
        # options="--opengl2"  #  --minGraphicsUpdateTimeMs=32000
        options=(
            "--background_color_red={}"
            " --background_color_green={}"
            " --background_color_blue={}"
        ).format(*background_color)
    )
    pybullet_path = pybullet_data.getDataPath()
    print("Adding pybullet data path {}".format(pybullet_path))
    pybullet.setAdditionalSearchPath(pybullet_path)


def real_time_handing(timestep, tic_rt, rtl=1.0, **kwargs):
    """Real-time handling"""
    sleep_rtl = timestep/rtl - (tic_rt[1] - tic_rt[0])
    rtf = timestep / (tic_rt[1] - tic_rt[0])
    tic = time.time()
    sleep_rtl = np.clip(sleep_rtl, a_min=0, a_max=1)
    if sleep_rtl > 0:
        while time.time() - tic < sleep_rtl:
            time.sleep(0.1*sleep_rtl)
    if rtf < 0.5:
        print("Significantly slower than real-time: {} %".format(100*rtf))
        time_plugin = kwargs.pop("time_plugin", False)
        time_control = kwargs.pop("time_control", False)
        time_sim = kwargs.pop("time_sim", False)
        if time_plugin:
            print("  Time in py_plugins: {} [ms]".format(time_plugin))
        if time_control:
            print("    Time in control: {} [ms]".format(time_control))
        if time_sim:
            print("  Time in simulation: {} [ms]".format(time_sim))
