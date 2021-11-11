"""Rendering"""

import pybullet


def rendering(render: int = 1):
    """Enable/disable rendering"""
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, render)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, render)
    # pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_TINY_RENDERER, render)
