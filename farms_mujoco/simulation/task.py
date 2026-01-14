"""Task"""

from typing import List, Dict

import numpy as np

from dm_control.rl.control import Task
from dm_control.viewer.application import Application
from dm_control.mjcf.physics import Physics
from dm_control.mujoco.wrapper import set_callback
from dm_control.mujoco.index import UnnamedAxis

from farms_core import pylog
from farms_core.extensions.extensions import import_item
from farms_core.simulation.extensions import TaskExtension
from farms_core.model.extensions import AnimatExtension
from farms_core.experiment.options import ExperimentOptions
from farms_core.model.control import ControlType, AnimatController
from farms_core.model.data import AnimatData
from farms_core.experiment.data import ExperimentData
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
from .mjcf import get_prefix


def duration2nit(duration: float, timestep: float) -> int:
    """Number of iterations from duration"""
    return int(duration/timestep)


class ExperimentTask(Task):
    """FARMS experiment"""

    def __init__(
            self,
            base_links: list[str],
            n_iterations: int,
            timestep: float,
            **kwargs,
    ):
        super().__init__()
        self.iteration: int = 0
        self.data: ExperimentData = kwargs.pop('data', None)
        self.viewer = kwargs.pop('viewer', None)
        self._app: Application | None = None
        self.timestep: float = timestep
        self.n_iterations: int = n_iterations
        self.mjcf = kwargs.pop('mjcf', None)
        self.base_links: list[str] = base_links  # TODO: Unused?
        if self.data is not None and not isinstance(self.data, ExperimentData):
            raise TypeError(
                'Data provided to ExperimentTask should be of type '
                f'ExperimentData, got {type(self.data)=} instead.'
            )
        self.experiment_options: ExperimentOptions = kwargs.pop('experiment_options')
        n_animats = len(self.experiment_options.animats)
        self._restart: bool = kwargs.pop('restart', True)
        self.extensions: List[TaskExtension] = self.extract_extensions(
            experiment_options=self.experiment_options,
            experiment_data=self.data,
        ) + kwargs.pop('extensions', [])
        self._extras: Dict = {'hfield': kwargs.pop('hfield', None)}
        self.units: SimulationUnits = kwargs.pop('units', SimulationUnits())
        self.buffer_size = max(1, kwargs.pop('buffer_size', 1))
        self.cb_sub_steps = max(1, kwargs.pop('substeps', 1))
        self.substeps_links = any(cb.substep for cb in self.extensions)
        self.sim_iteration = 0
        self.sim_iterations = self.n_iterations*self.cb_sub_steps
        self.sim_timestep = self.timestep/self.cb_sub_steps
        self.maps: list[Dict] = [{
            'sensors': {}, 'ctrl': {},
            'xpos': {}, 'qpos': {}, 'geoms': {},
            'links': {}, 'joints': {}, 'contacts': {}, 'xfrc': {},
            'muscles': {},
        } for _ in range(n_animats)]
        self.initialized = False
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

    @staticmethod
    def extract_extensions(experiment_options, experiment_data):
        """Extract extensiosn from experiment options"""

        # Simulation extensions
        simulation_extentions_loaders = [
            import_item(extension['loader'])
            for extension in experiment_options.simulation.extensions
        ]
        sim_extensions: list[TaskExtension] = [
            loader.from_options(
                config=extension['config'],
                experiment_options=experiment_options,
            )
            for loader, extension in zip(
                    simulation_extentions_loaders,
                    experiment_options.simulation.extensions
            )
        ]

        # Animat extensions
        animat_extensions: list[AnimatExtension] = [
            import_item(extension.loader).from_options(
                config=extension.config,
                experiment_options=experiment_options,
                animat_i=animat_i,
                animat_data=animat_data,
                animat_options=animat_options,
            )
            for animat_i, (animat_data, animat_options) in enumerate(zip(
                    experiment_data.animats,
                    experiment_options.animats,
            ))
            for extension in animat_options.extensions
        ]

        return sim_extensions + animat_extensions


    def initialize_episode(self, physics: Physics, viewer=None):
        """Sets the state of the environment at the start of each episode"""

        # Checks
        pylog.debug("Initializing episode")
        if self.initialized:
            pylog.warning('Simulation was already initialized')
        if self._restart:
            assert self._app is not None, (
                'Simulation can not be restarted without application interface'
            )

        # Viewer
        if viewer is not None:
            scn = viewer.user_scn
            scn.ngeom = 0

        # Links masses
        links_row = physics.named.model.body_mass.axes.row
        for animat_i, animat in enumerate(self.data.animats):
            prefix = get_prefix(animat_i)
            animat.sensors.links.masses = np.array([
                physics.model.body_mass[
                    links_row.convert_key_item(prefix+link_name)
                ]
                for link_name in animat.sensors.links.names
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
            if False and physics.contexts:
                # FIXME This leads to mujoco.FatalError: gladLoadGL error
                with physics.contexts.gl.make_current() as ctx:
                    ctx.call(
                        mjbindings.mjlib.mjr_uploadHField,
                        physics.model.ptr,
                        physics.contexts.mujoco.ptr,
                        physics.bind(hfield).element_id,
                    )
            if viewer is not None:
                viewer.update_hfield(physics.bind(hfield).element_id)

        # Maps, data and sensors
        self.initialize_maps(physics)
        if self.data is None:
            pylog.info('No data provided, initialising default')
            self.initialize_data()
        self.initialize_sensors(physics)

        # Controllers
        self._controllers: list[AnimatController] = [
            extension
            for extension in self.extensions
            if isinstance(extension, AnimatController)
        ]
        if self._controllers:
            self.initialize_control(physics)

        # Initialize joints to keyframe 0
        physics.reset(keyframe_id=0)

        if self._app is not None:
            self.update_sensors(physics, links_only=True)
            cam = self._app._viewer.camera  # pylint: disable=protected-access
            links = self.data.animats[0].sensors.links
            cam.look_at(
                position=links.urdf_position(iteration=0, link_i=0),
                distance=3,
            )

        if viewer is not None:
            self.update_sensors(physics, links_only=True)
            cam = viewer.cam
            links = self.data.animats[0].sensors.links
            cam.lookat = links.com_position(iteration=0, link_i=0)
            cam.azimuth = 70
            cam.elevation = -10

        # Extensions
        for extension in self.extensions:
            extension.initialize_episode(task=self, physics=physics)

        # Mujoco callbacks for muscle
        if rt_muscle:
            set_callback("mjcb_act_gain", rt_muscle.mjcb_muscle_gain)
            set_callback("mjcb_act_bias", rt_muscle.mjcb_muscle_bias)

        # Initialization complete
        self.initialized = True

    def update_sensors(self, physics: Physics, links_only=False):
        """Update sensors"""
        index = self.iteration % self.buffer_size
        self.data.times[index] = physics.time()
        sim_data = self.data.simulation
        sim_data.ncon[index] = physics.data.ncon
        sim_data.niter[index] = physics.data.solver_niter[0]
        sim_data.energy[index, :] = physics.data.energy
        for animat_i, animat_data in enumerate(self.data.animats):
            physics2data(
                physics=physics,
                iteration=index,
                data=animat_data,
                maps=self.maps[animat_i],
                units=self.units,
                links_only=links_only,
            )

    def before_step(self, action, physics: Physics):
        """Operations before physics step"""
        if physics.time() < 1e-6*self.sim_timestep:
            self.initialize_episode(physics, self.viewer)  # Reset

        full_step = not self.sim_iteration % self.cb_sub_steps

        # Checks
        assert self.iteration < self.n_iterations

        # Sensors
        if full_step or self.substeps_links:
            self.update_sensors(physics=physics, links_only=not full_step)

        # Extensions
        current_time = physics.time()
        index = self.iteration % self.buffer_size
        for extension in self.extensions:
            if full_step or extension.substep:
                extension.before_step(task=self, action=action, physics=physics)
                if isinstance(extension, AnimatController):
                    if extension.joints_names[ControlType.POSITION]:
                        self.step_joints_control_position(
                            controller=extension,
                            physics=physics,
                            time=current_time,
                        )
                    if extension.joints_names[ControlType.TORQUE]:
                        self.step_joints_control_torque(
                            controller=extension,
                            physics=physics,
                            time=current_time,
                        )
                    if extension.muscles_names:
                        muscles_excitations = extension.excitations(
                            iteration=index,
                            time=current_time,
                            timestep=self.timestep
                        )
                        muscle_indices = self.maps[animat_i]['ctrl']['mus']
                        physics.data.ctrl[muscle_indices] = muscles_excitations

    def initialize_maps(self, physics: Physics):
        """Initialise data"""
        physics_named = physics.named.data
        for sensor_map in self.maps:
            # Links indices
            sensor_map['xpos']['names'] = physics_named.xpos.axes.row.names
            # Joints indices
            sensor_map['qpos']['names'] = physics_named.qpos.axes.row.names
            # External forces indices
            sensor_map['xfrc']['names'] = physics_named.xfrc_applied.axes.row.names
            # Geoms indices
            sensor_map['geoms']['names'] = physics_named.geom_xpos.axes.row.names
            # Muscles indices
            # Check if any muscles present in the model
            if len(physics.model.tendon_adr) > 0:
                sensor_map['muscles']['names'] = physics_named.ten_length.axes.row.names
            else:
                sensor_map['muscles']['names'] = []

    def initialize_data(self):
        """Initialise data"""
        for animat_i, _ in enumerate(self.data.animats):
            self.data.animats[animat_i] = AnimatData.from_sensors_names(
                times=np.arange(
                    start=0,
                    stop=self.timestep*(self.n_iterations-0.5),
                    step=self.timestep,
                ),
                timestep=self.timestep,
                buffer_size=self.buffer_size,
                links=self.maps[animat_i]['xpos']['names'],
                joints=self.maps[animat_i]['qpos']['names'],
                muscles=self.maps[animat_i]['muscles']['names']
                # contacts=[],
                # xfrc=[],
            )

    def initialize_sensors(self, physics: Physics):
        """Initialise sensors"""
        for animat_i, animat in enumerate(self.data.animats):
            self.maps[animat_i]['sensors'] = get_sensor_maps(physics)
            prefix = get_prefix(animat_i)
            get_physics2data_maps(
                physics=physics,
                sensor_data=animat.sensors,
                sensor_maps=self.maps[animat_i]['sensors'],
                prefix=prefix,
            )

    def initialize_control(self, physics: Physics):
        """Initialise controller"""

        if isinstance(physics.named.data.ctrl.axes.row, UnnamedAxis):
            for controller in self._controllers:
                self.maps[controller.animat_i]['ctrl'] = None
            return

        ctrl_names = np.array(physics.named.data.ctrl.axes.row.names)
        for controller in self._controllers:
            prefix = get_prefix(controller.animat_i)
            for control_name, control_type in [
                    ('position', ControlType.POSITION),
                    ('velocity', ControlType.VELOCITY),
                    ('torque', ControlType.TORQUE),
            ]:
                for joint in controller.joints_names[control_type]:
                    assert (
                        f'actuator_{control_name}_{prefix}{joint}'
                        in
                        ctrl_names
                    ), (
                        f'Actuator for {joint} not in {ctrl_names}'
                    )

            # Joints maps
            self.maps[controller.animat_i]['ctrl']['pos'] = [
                np.argwhere(
                    ctrl_names
                    == f'actuator_position_{prefix}{joint}'
                )[0, 0]
                for joint in controller.joints_names[ControlType.POSITION]
            ]
            self.maps[controller.animat_i]['ctrl']['vel'] = [
                np.argwhere(
                    ctrl_names
                    == f'actuator_velocity_{prefix}{joint}'
                )[0, 0]
                for joint in controller.joints_names[ControlType.VELOCITY]
            ]
            self.maps[controller.animat_i]['ctrl']['trq'] = [
                np.argwhere(
                    ctrl_names
                    == f'actuator_torque_{prefix}{joint}'
                )[0, 0]
                for joint in controller.joints_names[ControlType.TORQUE]
            ]
            self.maps[controller.animat_i]['ctrl']['mus'] = [
                np.argwhere(ctrl_names == f'{prefix}{name}')[0, 0]
                for name in controller.muscles_names
            ]
            # Filter only actuated joints
            qpos_spring = physics.named.model.qpos_spring
            self.maps[controller.animat_i]['ctrl']['springref'] = {
                joint: qpos_spring.axes.row.convert_key_item(joint)
                for joint_i, joint in enumerate(qpos_spring.axes.row.names)
            }
            jnt_stiffness = physics.named.model.jnt_stiffness
            self.maps[controller.animat_i]['ctrl']['jnt_stiffness'] = {
                joint: jnt_stiffness.axes.row.convert_key_item(joint)
                for joint_i, joint in enumerate(jnt_stiffness.axes.row.names)
            }
            dof_damping = physics.named.model.dof_damping
            self.maps[controller.animat_i]['ctrl']['dof_damping'] = {
                joint: dof_damping.axes.row.convert_key_item(joint)
                for joint_i, joint in enumerate(dof_damping.axes.row.names)
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
        animats_options = self.experiment_options.animats
        for animat_i, animat_options in enumerate(animats_options):
            prefix = get_prefix(animat_i)
            for mtr_opts in animat_options.control.motors:
                jnt_name = mtr_opts['joint_name']
                if 'position' not in mtr_opts.control_types:
                    for act_type in ('pos', 'vel'):
                        if act_type in jntname2actid[prefix+jnt_name]:
                            physics.named.model.actuator_forcelimited[
                                jntname2actid[prefix+jnt_name][act_type]
                            ] = True
                            physics.named.model.actuator_forcerange[
                                jntname2actid[prefix+jnt_name][act_type]
                            ] = [0, 0]

    def step_joints_control_position(
            self,
            controller: AnimatController,
            physics: Physics,
            time: float,
    ):
        """Step position control"""
        index = self.iteration % self.buffer_size
        joints_positions = controller.positions(
            iteration=index,
            time=time,
            timestep=self.timestep,
        )
        physics.data.ctrl[self.maps[controller.animat_i]['ctrl']['pos']] = [
            joints_positions[joint]
            for joint in controller.joints_names[ControlType.POSITION]
        ]

    def step_joints_control_torque(
            self,
            controller: AnimatController,
            physics: Physics,
            time: float,
    ):
        """Step torque control"""
        index = self.iteration % self.buffer_size
        torques = self.units.torques
        animat_i = controller.animat_i
        prefix = get_prefix(animat_i)

        # Joints torques
        joints_torques = controller.torques(
            iteration=index,
            time=time,
            timestep=self.timestep,
        )
        physics.data.ctrl[self.maps[animat_i]['ctrl']['trq']] = [
            joints_torques[joint]*torques
            for joint in controller.joints_names[ControlType.TORQUE]
        ]

        # Spring reference
        qpos_spring = physics.model.qpos_spring
        springref_map = self.maps[animat_i]['ctrl']['springref']
        springrefs = controller.springrefs(
            iteration=index,
            time=time,
            timestep=self.timestep,
        )
        for joint, value in springrefs.items():
            qpos_spring[springref_map[prefix+joint]] = value

        # Spring coefs
        jnt_stiffness = physics.model.jnt_stiffness
        jnt_stiffness_map = self.maps[animat_i]['ctrl']['jnt_stiffness']
        springcoefs = controller.springcoefs(
            iteration=index,
            time=time,
            timestep=self.timestep,
        )
        for joint, value in springcoefs.items():
            jnt_stiffness[jnt_stiffness_map[prefix+joint]] = value

        # Dampings
        dof_damping = physics.model.dof_damping
        dof_damping_map = self.maps[animat_i]['ctrl']['dof_damping']
        dampingcoefs = controller.dampingcoefs(
            iteration=index,
            time=time,
            timestep=self.timestep,
        )
        for joint, value in dampingcoefs.items():
            dof_damping[dof_damping_map[prefix+joint]] = value

    def after_step(self, physics: Physics):
        """Operations after physics step"""

        # Checks
        self.sim_iteration += 1
        fullstep = not (self.sim_iteration + 1) % self.cb_sub_steps
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

        # Extensions
        if fullstep:
            for extension in self.extensions:
                extension.after_step(task=self, physics=physics)

    def action_spec(self, physics: Physics):
        """Action specifications"""
        specs = []
        for extension in self.extensions:
            spec = extension.action_spec(task=self, physics=physics)
            if spec is not None:
                specs += spec
        return specs

    def step_spec(self, physics: Physics):
        """Timestep specifications"""
        for extension in self.extensions:
            extension.step_spec(task=self, physics=physics)

    def get_observation(self, physics: Physics):
        """Environment observation"""
        for extension in self.extensions:
            extension.get_observation(task=self, physics=physics)

    def get_reward(self, physics: Physics):
        """Reward"""
        reward = 0
        for extension in self.extensions:
            extension_reward = extension.get_reward(task=self, physics=physics)
            if extension_reward is not None:
                reward += extension_reward
        return reward

    def get_termination(self, physics: Physics):
        """Return final discount if episode should end, else None"""
        terminate = None
        for extension in self.extensions:
            if extension.get_termination(task=self, physics=physics):
                terminate = 1
        if self.iteration >= self.n_iterations:
            terminate = 1
        return terminate

    def observation_spec(self, physics: Physics):
        """Observation specifications"""
        for extension in self.extensions:
            extension.observation_spec(task=self, physics=physics)
