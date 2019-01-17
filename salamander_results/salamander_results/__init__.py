""" Salamander results """

__all__ = [
    "extract_logs",
    "extract_positions",
    "extract_consumption",
    "extract_final_consumption"
    # "plot_links_positions",
    # "plot_models_positions",
    # "plot_joints_positions",
    # "plot_joints_cmd_pos",
    # "plot_joints_cmd_vel",
    # "plot_joints_cmd_torque",
    # "plot_joints_cmd_consumption",
]

from .extract import (
    extract_logs,
    extract_positions,
    extract_consumption,
    extract_final_consumption
)

# from .plot import (
#     plot_links_positions,
#     plot_models_positions,
#     plot_joints_positions,
#     plot_joints_cmd_pos,
#     plot_joints_cmd_vel,
#     plot_joints_cmd_torque,
#     plot_joints_cmd_consumption
# )
