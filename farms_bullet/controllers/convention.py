"""Oscillator naming convention"""

def bodyosc2index(joint_i, side=0, n_body_joints=11):
    """body2index"""
    return joint_i + side*n_body_joints

def legosc2index(
        leg_i, side_i, joint_i, side=0, n_body_joints=11, n_legs_dof=3
):
    """legjoint2index"""
    return (
        2*n_body_joints
        + leg_i*2*n_legs_dof*2
        + side_i*n_legs_dof*2
        + joint_i
        + side*n_legs_dof
    )

def legjoint2index(
        leg_i, side_i, joint_i, n_body_joints=11, n_legs_dof=3
):
    """legjoint2index"""
    return (
        n_body_joints
        + leg_i*2*n_legs_dof
        + side_i*n_legs_dof
        + joint_i
    )
