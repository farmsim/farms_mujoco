"""Control"""

import pybullet


class AnimatController:
    """AnimatController"""

    def __init__(self, model, network, units):
        super(AnimatController, self).__init__()
        self.model = model
        self.network = network
        self.positions = None
        self.velocities = None
        self.torques = None
        self.iseconds = 1./units.seconds

    def update(self):
        """Step"""
        self.network.control_step()
        self.positions = self.network.get_position_output()
        self.velocities = self.network.get_velocity_output()

    def control(self):
        """Control"""
        self.update()
        pybullet.setJointMotorControlArray(
            self.model,
            range(11+4*4),
            pybullet.POSITION_CONTROL,
            targetPositions=self.positions,
            targetVelocities=self.velocities*self.iseconds,
            # positionGains=[ctrl["pdf"]["p"] for ctrl in controls],
            # velocityGains=[ctrl["pdf"]["d"] for ctrl in controls],
            # forces=[ctrl["pdf"]["f"] for ctrl in controls]
        )
