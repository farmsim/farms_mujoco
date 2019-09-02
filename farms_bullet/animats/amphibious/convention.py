"""Oscillator naming convention"""

class AmphibiousConvention:
    """Amphibious convention"""

    def __init__(self, animat_options):
        super(AmphibiousConvention, self).__init__()
        self.animat_options = animat_options

    def bodyosc2index(self, joint_i, side=0):
        """body2index"""
        n_body_joints = self.animat_options.morphology.n_joints_body
        assert 0 <= joint_i < n_body_joints, "Joint must be < {}, got {}".format(
            n_body_joints,
            joint_i
        )
        return joint_i + side*n_body_joints

    def legosc2index(self, leg_i, side_i, joint_i, side=0):
        """legosc2index"""
        n_legs = self.animat_options.morphology.n_legs
        n_body_joints = self.animat_options.morphology.n_joints_body
        n_legs_dof = self.animat_options.morphology.n_dof_legs
        assert 0 <= leg_i < n_legs, "Leg must be < {}, got {}".format(n_legs//2, leg_i)
        assert 0 <= side_i < 2, "Body side must be < 2, got {}".format(side_i)
        assert 0 <= joint_i < n_legs_dof, "Joint must be < {}, got {}".format(n_legs_dof, joint_i)
        assert 0 <= side < 2, "Oscillator side must be < 2, got {}".format(side)
        return (
            2*n_body_joints
            + leg_i*2*n_legs_dof*2  # 2 oscillators, 2 legs
            + side_i*n_legs_dof*2  # 2 oscillators
            + joint_i
            + side*n_legs_dof
        )

    def leglink2index(self, leg_i, side_i, joint_i):
        """leglink2index"""
        n_legs = self.animat_options.morphology.n_legs
        n_body_links = self.animat_options.morphology.n_links_body()
        n_legs_dof = self.animat_options.morphology.n_dof_legs
        assert 0 <= leg_i < n_legs, "Leg must be < {}, got {}".format(n_legs//2, leg_i)
        assert 0 <= side_i < 2, "Body side must be < 2, got {}".format(side_i)
        assert 0 <= joint_i < n_legs_dof, "Joint must be < {}, got {}".format(n_legs_dof, joint_i)
        return (
            n_body_links - 1
            + leg_i*2*n_legs_dof
            + side_i*n_legs_dof
            + joint_i
        )

    def leglink2name(self, leg_i, side_i, joint_i):
        """leglink2index"""
        n_legs = self.animat_options.morphology.n_legs
        n_legs_dof = self.animat_options.morphology.n_dof_legs
        assert 0 <= leg_i < n_legs, "Leg must be < {}, got {}".format(n_legs//2, leg_i)
        assert 0 <= side_i < 2, "Body side must be < 2, got {}".format(side_i)
        assert 0 <= joint_i < n_legs_dof, "Joint must be < {}, got {}".format(n_legs_dof, joint_i)
        return "link_leg_{}_{}_{}".format(leg_i, "R" if side_i else "L", joint_i)

    def legjoint2index(self, leg_i, side_i, joint_i):
        """legjoint2index"""
        n_legs = self.animat_options.morphology.n_legs
        n_body_joints = self.animat_options.morphology.n_joints_body
        n_legs_dof = self.animat_options.morphology.n_dof_legs
        assert 0 <= leg_i < n_legs, "Leg must be < {}, got {}".format(n_legs//2, leg_i)
        assert 0 <= side_i < 2, "Body side must be < 2, got {}".format(side_i)
        assert 0 <= joint_i < n_legs_dof, "Joint must be < {}, got {}".format(n_legs_dof, joint_i)
        return (
            n_body_joints
            + leg_i*2*n_legs_dof
            + side_i*n_legs_dof
            + joint_i
        )

    def legjoint2name(self, leg_i, side_i, joint_i):
        """legjoint2index"""
        n_legs = self.animat_options.morphology.n_legs
        n_legs_dof = self.animat_options.morphology.n_dof_legs
        assert 0 <= leg_i < n_legs, "Leg must be < {}, got {}".format(n_legs//2, leg_i)
        assert 0 <= side_i < 2, "Body side must be < 2, got {}".format(side_i)
        assert 0 <= joint_i < n_legs_dof, "Joint must be < {}, got {}".format(n_legs_dof, joint_i)
        return "joint_{}".format(
            self.leglink2name(
                leg_i,
                side_i,
                joint_i
            )
        )

    def contactleglink2index(self, leg_i, side_i):
        """Contact leg link 2 index"""
        n_legs = self.animat_options.morphology.n_legs
        assert 0 <= leg_i < n_legs, "Leg must be < {}, got {}".format(n_legs//2, leg_i)
        assert 0 <= side_i < 2, "Body side must be < 2, got {}".format(side_i)
        return 2*leg_i + side_i
