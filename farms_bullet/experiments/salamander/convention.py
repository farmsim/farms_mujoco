"""Oscillator naming convention"""

def bodyosc2index(joint_i, side=0, n_body_joints=11):
    """body2index"""
    assert joint_i < 11, "Joint must be < 11, got {}".format(joint_i)
    return joint_i + side*n_body_joints

def legosc2index(leg_i, side_i, joint_i, side=0, **kwargs):
    """legosc2index"""
    n_body_joints = kwargs.pop("n_body_joints", 11)
    n_legs_dof = kwargs.pop("n_legs_dof", 4)
    return (
        2*n_body_joints
        + leg_i*2*n_legs_dof*2  # 2 oscillators, 2 legs
        + side_i*n_legs_dof*2  # 2 oscillators
        + joint_i
        + side*n_legs_dof
    )

def legjoint2index(leg_i, side_i, joint_i, n_body_joints=11, n_legs_dof=4):
    """legjoint2index"""
    return (
        n_body_joints
        + leg_i*2*n_legs_dof
        + side_i*n_legs_dof
        + joint_i
    )
