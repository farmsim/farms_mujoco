"""Oscillator naming convention"""

def bodyosc2index(joint_i, side=0, n_body_joints=11):
    """body2index"""
    assert 0 <= joint_i < 11, "Joint must be < 11, got {}".format(joint_i)
    return joint_i + side*n_body_joints


def legosc2index(leg_i, side_i, joint_i, side=0, **kwargs):
    """legosc2index"""
    n_legs = kwargs.pop("n_legs", 4)
    n_body_joints = kwargs.pop("n_body_joints", 11)
    n_legs_dof = kwargs.pop("n_legs_dof", 4)
    assert 0 <= leg_i < 2, "Leg must be < {}, got {}".format(n_legs//2, leg_i)
    assert 0 <= side_i < 2, "Body side must be < 2, got {}".format(side_i)
    assert 0 <= joint_i < 4, "Joint must be < {}, got {}".format(n_legs_dof, joint_i)
    assert 0 <= side < 2, "Oscillator side must be < 2, got {}".format(side)
    return (
        2*n_body_joints
        + leg_i*2*n_legs_dof*2  # 2 oscillators, 2 legs
        + side_i*n_legs_dof*2  # 2 oscillators
        + joint_i
        + side*n_legs_dof
    )


def leglink2index(leg_i, side_i, joint_i, **kwargs):
    """leglink2index"""
    n_legs = kwargs.pop("n_legs", 4)
    n_body_links = kwargs.pop("n_body_links", 12)
    n_legs_dof = kwargs.pop("n_legs_dof", 4)
    assert 0 <= leg_i < 2, "Leg must be < {}, got {}".format(n_legs//2, leg_i)
    assert 0 <= side_i < 2, "Body side must be < 2, got {}".format(side_i)
    assert 0 <= joint_i < 4, "Joint must be < {}, got {}".format(n_legs_dof, joint_i)
    return (
        n_body_links - 1
        + leg_i*2*n_legs_dof
        + side_i*n_legs_dof
        + joint_i
    )


def leglink2name(leg_i, side_i, joint_i, **kwargs):
    """leglink2index"""
    n_legs = kwargs.pop("n_legs", 4)
    n_legs_dof = kwargs.pop("n_legs_dof", 4)
    assert 0 <= leg_i < 2, "Leg must be < {}, got {}".format(n_legs//2, leg_i)
    assert 0 <= side_i < 2, "Body side must be < 2, got {}".format(side_i)
    assert 0 <= joint_i < 4, "Joint must be < {}, got {}".format(n_legs_dof, joint_i)
    return "link_leg_{}_{}_{}".format(leg_i, "R" if side_i else "L", joint_i)


def legjoint2index(leg_i, side_i, joint_i, **kwargs):
    """legjoint2index"""
    n_legs = kwargs.pop("n_legs", 4)
    n_body_joints = kwargs.pop("n_body_joints", 11)
    n_legs_dof = kwargs.pop("n_legs_dof", 4)
    assert 0 <= leg_i < 2, "Leg must be < {}, got {}".format(n_legs//2, leg_i)
    assert 0 <= side_i < 2, "Body side must be < 2, got {}".format(side_i)
    assert 0 <= joint_i < 4, "Joint must be < {}, got {}".format(n_legs_dof, joint_i)
    return (
        n_body_joints
        + leg_i*2*n_legs_dof
        + side_i*n_legs_dof
        + joint_i
    )


def legjoint2name(leg_i, side_i, joint_i, **kwargs):
    """legjoint2index"""
    n_legs = kwargs.pop("n_legs", 4)
    n_legs_dof = kwargs.pop("n_legs_dof", 4)
    assert 0 <= leg_i < 2, "Leg must be < {}, got {}".format(n_legs//2, leg_i)
    assert 0 <= side_i < 2, "Body side must be < 2, got {}".format(side_i)
    assert 0 <= joint_i < 4, "Joint must be < {}, got {}".format(n_legs_dof, joint_i)
    return "joint_{}".format(leglink2name(leg_i, side_i, joint_i))


def contactleglink2index(leg_i, side_i, **kwargs):
    """Contact leg link 2 index"""
    n_legs = kwargs.pop("n_legs", 4)
    assert 0 <= leg_i < 2, "Leg must be < {}, got {}".format(n_legs//2, leg_i)
    assert 0 <= side_i < 2, "Body side must be < 2, got {}".format(side_i)
    return 2*leg_i + side_i
