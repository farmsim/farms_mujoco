"""Simulation"""

import pybullet
from tqdm import tqdm

import farms_pylog as pylog

from ..model.control import control_models
from .simulator import init_engine
from .render import rendering


def simulation_profiler(func):
    """Profile simulation"""
    def inner(self, profile=False, show_progress=False):
        """Inner function"""
        if profile:
            logger = pybullet.startStateLogging(
                loggingType=pybullet.STATE_LOGGING_PROFILE_TIMINGS,
                fileName='profile.log'
            )
        pbar = tqdm(total=self.options.n_iterations) if show_progress else None
        result = func(self, pbar=pbar)
        if profile:
            pybullet.stopStateLogging(loggingId=logger)
        return result
    return inner


class Simulation:
    """Simulation

    Handles the start/run/end of the experiment, the GUI if not headless, the
    physics properties, etc.

    """

    def __init__(self, models, options):
        super(Simulation, self).__init__()

        self.models = models
        self.options = options

        # Initialise engine
        pylog.debug('Initialising physics engine')
        init_engine(self.options.headless, self.options.opengl2)
        if not self.options.headless:
            pylog.debug('Disabling rendering')
            rendering(0)

        # Initialise physics
        pylog.debug('Initialising physics')
        self.init_physics()

        # Initialise models
        pylog.debug('Spawning models')
        self.models.spawn()

        # Simulation
        self.iteration = 0
        self.simulation_state = None

        # Interface
        self.interface = None

        if not self.options.headless:
            pylog.debug('Reactivating rendering')
            rendering(1)

    def save(self):
        """Save experiment state"""
        self.simulation_state = pybullet.saveState()

    def init_physics(self):
        """Initialise physics"""
        # pylog.debug('Resetting simulation')
        # pybullet.resetSimulation()
        pylog.debug('Setting gravity')
        pybullet.setGravity(
            self.options.gravity[0]*self.options.units.gravity,
            self.options.gravity[1]*self.options.units.gravity,
            self.options.gravity[2]*self.options.units.gravity
        )
        pylog.debug('Setting timestep')
        pybullet.setTimeStep(self.options.timestep*self.options.units.seconds)
        pylog.debug('Setting non real-time simulation')
        pybullet.setRealTimeSimulation(0)
        pylog.debug('Setting simulation parameters')
        pybullet.setPhysicsEngineParameter(
            fixedTimeStep=self.options.timestep*self.options.units.seconds,
            numSolverIterations=self.options.n_solver_iters,
            erp=self.options.erp,
            contactERP=self.options.contact_erp,
            frictionERP=self.options.friction_erp,
            numSubSteps=self.options.num_sub_steps,
            maxNumCmdPer1ms=self.options.max_num_cmd_per_1ms,
            solverResidualThreshold=self.options.residual_threshold,
            # solverResidualThreshold=1e-12,
            # restitutionVelocityThreshold=1e-3,
            # useSplitImpulse=False,
            # splitImpulsePenetrationThreshold=1e-5,
            # contactBreakingThreshold=1e-5
            # numSubSteps=100,
            # maxNumCmdPer1ms=int(1e5),

            # # Parameters
            # fixedTimeStep
            # numSolverIterations
            # useSplitImpulse
            # splitImpulsePenetrationThreshold
            # numSubSteps
            # collisionFilterMode
            # contactBreakingThreshold
            # maxNumCmdPer1ms
            # enableFileCaching
            # restitutionVelocityThreshold
            # erp
            # contactERP
            # frictionERP
            # enableConeFriction
            # deterministicOverlappingPairs
            # solverResidualThreshold
        )
        pylog.debug('Physics parameters:\n{}'.format(
            pybullet.getPhysicsEngineParameters()
        ))

    def check_quit(self):
        """Check quit"""
        if not self.options.headless:
            keys = pybullet.getKeyboardEvents()
            if ord('q') in keys:
                return True
        return False

    @simulation_profiler
    def run(self, pbar=None):
        """Run simulation"""
        while self.iteration < self.options.n_iterations:
            if self.check_quit():
                break
            if self.pre_step(self.iteration):
                self.step(self.iteration)
                self.control(self.iteration)
                pybullet.stepSimulation()
                self.iteration += 1
                self.post_step(self.iteration)
                if pbar is not None:
                    pbar.update(1)

    @simulation_profiler
    def iterator(self, pbar=None):
        """Run simulation"""
        while self.iteration < self.options.n_iterations:
            if self.check_quit():
                break
            if self.pre_step(self.iteration):
                self.step(self.iteration)
                self.control(self.iteration)
                pybullet.stepSimulation()
                self.iteration += 1
                self.post_step(self.iteration)
                yield self.iteration-1
                if pbar is not None:
                    pbar.update(1)

    def pre_step(self, iteration):
        """Pre-step

        Returns bool

        """

    def step(self, iteration):
        """Step function"""

    def control(self, iteration):
        """Physics step"""
        control_models(
            iteration=iteration,
            models=self.models,
            torques=self.options.units.torques,
        )

    def post_step(self, iteration):
        """Post-step"""

    def end(self):
        """Terminate simulation"""
        # Disconnect from simulation
        pybullet.disconnect()
