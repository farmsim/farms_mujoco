""" Salamander results """

__all__ = [
    "extract_logs",
    "plot_links_positions",
    "plot_models_positions",
    "plot_joints_positions",
    "plot_joints_cmd_pos",
    "plot_joints_cmd_vel",
    "plot_joints_cmd_torque",
    "plot_joints_cmd_consumption",
    "extract_consumption",
    "extract_final_consumption"
]

from .plot import (
    extract_logs,
    plot_links_positions,
    plot_models_positions,
    plot_joints_positions,
    plot_joints_cmd_pos,
    plot_joints_cmd_vel,
    plot_joints_cmd_torque,
    plot_joints_cmd_consumption,
    extract_consumption,
    extract_final_consumption
)
