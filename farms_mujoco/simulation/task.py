"""Task"""

from typing import List, Dict

import numpy as np

from dm_control.rl.control import Task
from dm_control.mujoco.wrapper import mjbindings
from dm_control.viewer.application import Application
from dm_control.mjcf.physics import Physics
from dm_control.mujoco.wrapper import set_callback

from farms_core import pylog
from farms_core.model.options import AnimatOptions
from farms_core.model.control import ControlType, AnimatController
from farms_core.model.data import AnimatData
from farms_core.units import SimulationUnitScaling as SimulationUnits

try:
    from farms_muscle import rigid_tendon as rt_muscle
except:
    rt_muscle = None
    pylog.warning("farms_muscle not installed!")

from .physics import (
    get_sensor_maps,
    get_physics2data_maps,
    physics2data,
)


def duration2nit(duration: float, timestep: float) -> int:
    """Number of iterations from duration"""
    return int(duration/timestep)


class ExperimentTask(Task):
    """FARMS experiment"""

    def __init__(
            self,
            base_link: str,
            n_iterations: int,
            timestep: float,
            **kwargs,
    ):
        super().__init__()
        self._app: Application = None
        self.iteration: int = 0
        self.timestep: float = timestep
        self.n_iterations: int = n_iterations
        self.base_link: str = base_link
        self.data: AnimatData = kwargs.pop('data', None)
        self._controller: AnimatController = kwargs.pop('controller', None)
        self.animat_options: AnimatOptions = kwargs.pop('animat_options', None)
        self.external_force: float = kwargs.pop('external_force', 0.2)
        self._restart: bool = kwargs.pop('restart', True)
        self._callbacks: List[TaskCallback] = kwargs.pop('callbacks', [])
        self._extras: Dict = {'hfield': kwargs.pop('hfield', None)}
        self.units: SimulationUnits = kwargs.pop('units', SimulationUnits())
        self.substeps = max(1, kwargs.pop('substeps', 1))
        self.substeps_links = any(cb.substep for cb in self._callbacks)
        self.sim_iteration = 0
        self.sim_iterations = self.n_iterations*self.substeps
        self.sim_timestep = self.timestep/self.substeps
        self.maps: Dict = {
            'sensors': {}, 'ctrl': {},
            'xpos': {}, 'qpos': {}, 'geoms': {},
            'links': {}, 'joints': {}, 'contacts': {}, 'xfrc': {},
            'muscles': {}
        }
        assert not kwargs, kwargs

    def __del__(self):
        """ Destructor """
        # It is necessary to remove the callbacks to avoid crashes in
        # mujoco reruns
        set_callback("mjcb_act_gain", None)
        set_callback("mjcb_act_bias", None)

    def set_app(self, app: Application):
        """Set application"""
        assert isinstance(app, Application)
        self._app = app

    def initialize_episode(self, physics: Physics):
        """Sets the state of the environment at the start of each episode"""

        # Checks
        if self._restart:
            assert self._app is not None, (
                'Simulation can not be restarted without application interface'
            )

        # Links masses
        links_row = physics.named.model.body_mass.axes.row
        self.data.sensors.links.masses = np.array([
            physics.model.body_mass[links_row.convert_key_item(link_name)]
            for link_name in self.data.sensors.links.names
        ], dtype=float)/self.units.kilograms

        # Initialise iterations
        self.iteration = 0
        self.sim_iteration = 0

        # Initialise terrain
        if self._extras['hfield'] is not None:
            data = self._extras['hfield']['data']
            hfield = self._extras['hfield']['asset']
            nrow = physics.bind(hfield).nrow
            ncol = physics.bind(hfield).ncol
            idx0 = physics.bind(hfield).adr
            size = nrow*ncol
            physics.model.hfield_data[idx0:idx0+size] = 2*(data.flatten()-0.5)
            if physics.contexts:
                with physics.contexts.gl.make_current() as ctx:
                    ctx.call(
                        mjbindings.mjlib.mjr_uploadHField,
                        physics.model.ptr,
                        physics.contexts.mujoco.ptr,
                        physics.bind(hfield).element_id,
                    )

        # Maps, data and sensors
        self.initialize_maps(physics)
        if self.data is None:
            pylog.info('No data provided, initialising default')
            self.initialize_data()
        self.initialize_sensors(physics)

        # Control
        if self._controller is not None:
            self.initialize_control(physics)

        # Intitialize base link
        if not self.animat_options.mujoco.get('fixed_base', False):
            physics.data.qvel[:6] = self.animat_options.spawn.velocity

        # Initialize joints to keyframe 0
        physics.reset(keyframe_id=0)

        if self._app is not None:
            cam = self._app._viewer.camera  # pylint: disable=protected-access
            links = self.data.sensors.links
            cam.look_at(
                position=links.urdf_position(iteration=0, link_i=0),
                distance=3,
            )

        # Callbacks
        for callback in self._callbacks:
            callback.initialize_episode(task=self, physics=physics)

        # Mujoco callbacks for muscle
        if rt_muscle:
            set_callback("mjcb_act_gain", rt_muscle.mjcb_muscle_gain)
            set_callback("mjcb_act_bias", rt_muscle.mjcb_muscle_bias)

    def update_sensors(self, physics: Physics, links_only=False):
        """Update sensors"""
        physics2data(
            physics=physics,
            iteration=self.iteration,
            data=self.data,
            maps=self.maps,
            units=self.units,
            links_only=links_only,
        )

    def before_step(self, action, physics: Physics):
        """Operations before physics step"""

        # Checks
        assert self.iteration < self.n_iterations

        # Sensors
        full_step = not self.sim_iteration % self.substeps
        if full_step or self.substeps_links:
            self.update_sensors(physics=physics, links_only=not full_step)

        # Callbacks
        for callback in self._callbacks:
            if full_step or callback.substep:
                callback.before_step(task=self, action=action, physics=physics)

        # Control
        if full_step and self._controller is not None:
            self.step_control(physics)

    def initialize_maps(self, physics: Physics):
        """Initialise data"""
        physics_named = physics.named.data
        # Links indices
        self.maps['xpos']['names'] = physics_named.xpos.axes.row.names
        # Joints indices
        self.maps['qpos']['names'] = physics_named.qpos.axes.row.names
        # External forces indices
        self.maps['xfrc']['names'] = physics_named.xfrc_applied.axes.row.names
        # Geoms indices
        self.maps['geoms']['names'] = physics_named.geom_xpos.axes.row.names
        # Muscles indices
        # Check if any muscles present in the model
        if len(physics.model.tendon_adr) > 0:
            self.maps['muscles']['names'] = physics_named.ten_length.axes.row.names
        else:
            self.maps['muscles']['names'] = []

    def initialize_data(self):
        """Initialise data"""
        self.data = AnimatData.from_sensors_names(
            timestep=self.timestep,
            n_iterations=self.n_iterations,
            links=self.maps['xpos']['names'],
            joints=self.maps['qpos']['names'],
            muscles=self.maps['muscles']['names']
            # contacts=[],
            # xfrc=[],
        )

    def initialize_sensors(self, physics: Physics):
        """Initialise sensors"""
        self.maps['sensors'] = get_sensor_maps(physics)
        get_physics2data_maps(
            physics=physics,
            sensor_data=self.data.sensors,
            sensor_maps=self.maps['sensors'],
        )

    def initialize_control(self, physics: Physics):
        """Initialise controller"""
        ctrl_names = np.array(physics.named.data.ctrl.axes.row.names)
        for joint in self._controller.joints_names[ControlType.POSITION]:
            assert f'actuator_position_{joint}' in ctrl_names, (
                f'{joint} not in {ctrl_names}'
            )

        # Joints maps
        self.maps['ctrl']['pos'] = [
            np.argwhere(ctrl_names == f'actuator_position_{joint}')[0, 0]
            for joint in self._controller.joints_names[ControlType.POSITION]
        ]
        self.maps['ctrl']['vel'] = [
            np.argwhere(ctrl_names == f'actuator_velocity_{joint}')[0, 0]
            for joint in self._controller.joints_names[ControlType.VELOCITY]
        ]
        self.maps['ctrl']['trq'] = [
            np.argwhere(ctrl_names == f'actuator_torque_{joint}')[0, 0]
            for joint in self._controller.joints_names[ControlType.TORQUE]
        ]
        if self._controller.muscles_names:
            self.maps['ctrl']['mus'] = [
                np.argwhere(ctrl_names == f'{name}')[0, 0]
                for name in self._controller.muscles_names
            ]
        # Filter only actuated joints
        qpos_spring = physics.named.model.qpos_spring
        self.maps['ctrl']['springref'] = {
            joint: qpos_spring.axes.row.convert_key_item(joint)
            for joint_i, joint in enumerate(qpos_spring.axes.row.names)
        }
        act_trnid = physics.named.model.actuator_trnid
        act_trntype = physics.named.model.actuator_trntype
        jnt_names = physics.named.model.jnt_type.axes.row.names
        jntname2actid = {name: {} for name in jnt_names}
        for act_i, act_bias in enumerate(physics.model.actuator_biasprm):
            if act_trntype[act_i] < 2:
                act_type = (
                    'pos' if act_bias[1] != 0
                    else 'vel' if act_bias[2] != 0
                    else 'trq'
                )
                jnt_name = jnt_names[act_trnid[act_i][0]]
                jntname2actid[jnt_name][act_type] = act_i

        # Actuator limits
        if self.animat_options is not None:
            animat_options = self.animat_options
            for mtr_opts in animat_options.control.motors:
                jnt_name = mtr_opts['joint_name']
                if 'position' not in mtr_opts.control_types:
                    for act_type in ('pos', 'vel'):
                        if act_type in jntname2actid[jnt_name]:
                            physics.named.model.actuator_forcelimited[
                                jntname2actid[jnt_name][act_type]
                            ] = True
                            physics.named.model.actuator_forcerange[
                                jntname2actid[jnt_name][act_type]
                            ] = [0, 0]

    def step_control(self, physics: Physics):
        """Step control"""
        current_time = self.iteration*self.timestep
        self._controller.step(
            iteration=self.iteration,
            time=current_time,
            timestep=self.timestep,
        )
        if self._controller.joints_names[ControlType.POSITION]:
            self.step_joints_control_position(physics, current_time)
        if self._controller.joints_names[ControlType.TORQUE]:
            self.step_joints_control_torque(physics, current_time)
        if self._controller.muscles_names:
            muscles_excitations = self._controller.excitations(
                iteration=self.iteration,
                time=current_time,
                timestep=self.timestep
            )
            physics.data.ctrl[self.maps['ctrl']['mus']] = muscles_excitations

    def step_joints_control_position(self, physics: Physics, time: float):
        """Step position control"""
        joints_positions = self._controller.positions(
            iteration=self.iteration,
            time=time,
            timestep=self.timestep,
        )
        physics.data.ctrl[self.maps['ctrl']['pos']] = [
            joints_positions[joint]
            for joint
            in self._controller.joints_names[ControlType.POSITION]
        ]

    def step_joints_control_torque(self, physics: Physics, time: float):
        """Step torque control"""
        joints_torques = self._controller.torques(
            iteration=self.iteration,
            time=time,
            timestep=self.timestep,
        )
        torques = self.units.torques
        physics.data.ctrl[self.maps['ctrl']['trq']] = [
            joints_torques[joint]*torques
            for joint
            in self._controller.joints_names[ControlType.TORQUE]
        ]
        # Spring reference
        springrefs = self._controller.springrefs(
            iteration=self.iteration,
            time=time,
            timestep=self.timestep,
        )
        qpos_spring = physics.model.qpos_spring
        springref_map = self.maps['ctrl']['springref']
        for joint, value in springrefs.items():
            qpos_spring[springref_map[joint]] = value

    def after_step(self, physics: Physics):
        """Operations after physics step"""

        # Checks
        self.sim_iteration += 1
        fullstep = not (self.sim_iteration + 1) % self.substeps
        if fullstep:
            self.iteration += 1
        assert self.iteration <= self.n_iterations

        # Simulation complete
        if self.iteration == self.n_iterations:
            pylog.info('Simulation complete')
            if self._app is not None and not self._restart:
                self._app.close()
            else:
                pylog.info('Simulation can be restarted')

        # Callbacks
        if fullstep:
            for callback in self._callbacks:
                callback.after_step(task=self, physics=physics)

    def action_spec(self, physics: Physics):
        """Action specifications"""
        specs = []
        for callback in self._callbacks:
            spec = callback.action_spec(task=self, physics=physics)
            if spec is not None:
                specs += spec
        return specs

    def step_spec(self, physics: Physics):
        """Timestep specifications"""
        for callback in self._callbacks:
            callback.step_spec(task=self, physics=physics)

    def get_observation(self, physics: Physics):
        """Environment observation"""
        for callback in self._callbacks:
            callback.get_observation(task=self, physics=physics)

    def get_reward(self, physics: Physics):
        """Reward"""
        reward = 0
        for callback in self._callbacks:
            callback_reward = callback.get_reward(task=self, physics=physics)
            if callback_reward is not None:
                reward += callback_reward
        return reward

    def get_termination(self, physics: Physics):
        """Return final discount if episode should end, else None"""
        terminate = None
        for callback in self._callbacks:
            if callback.get_termination(task=self, physics=physics):
                terminate = 1
        if self.iteration >= self.n_iterations:
            terminate = 1
        return terminate

    def observation_spec(self, physics: Physics):
        """Observation specifications"""
        for callback in self._callbacks:
            callback.observation_spec(task=self, physics=physics)


class TaskCallback:
    """Task callback"""

    def __init__(self, substep=False):
        self.substep = substep

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        """Initialize episode"""

    def before_step(self, task: ExperimentTask, action, physics: Physics):
        """Before step"""

    def after_step(self, task: ExperimentTask, physics: Physics):
        """After step"""

    def action_spec(self, task: ExperimentTask, physics: Physics):
        """Action specifications"""

    def step_spec(self, task: ExperimentTask, physics: Physics):
        """Timestep specifications"""

    def get_observation(self, task: ExperimentTask, physics: Physics):
        """Environment observation"""

    def get_reward(self, task: ExperimentTask, physics: Physics):
        """Reward"""

    def get_termination(self, task: ExperimentTask, physics: Physics):
        """Return final discount if episode should end, else None"""

    def observation_spec(self, task: ExperimentTask, physics: Physics):
        """Observation specifications"""
