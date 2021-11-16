"""MJCF handling"""

import os
import xml.dom.minidom
import xml.etree.ElementTree as ET

import numpy as np
import trimesh as tri
from scipy.spatial.transform import Rotation

from dm_control import mjcf

import farms_pylog as pylog
from farms_data.units import SimulationUnitScaling
from farms_sdf.sdf import (
    ModelSDF, Mesh, Visual, Collision,
    Box, Cylinder, Capsule, Sphere, Plane,
)

MIN_MASS = 1e-12
MIN_INERTIA = 1e-12


def euler2mjcquat(euler):
    """Euler to Mujoco quaternion"""
    return Rotation.from_euler(
        angles=euler,
        seq='xyz',
    ).as_quat()[[3, 0, 1, 2]]


def euler2mat(euler):
    """Euler to Mujoco quaternion"""
    return Rotation.from_euler(
        angles=euler,
        seq='xyz',
    ).as_matrix()


def poseul2mat4d(position, euler):
    """4D transform"""
    transform = np.eye(4)
    transform[:3, -1] = position
    transform[:3, :3] = euler2mat(euler)
    return transform


def get_local_transform(parent_pose, child_pose):
    """Get link local transform"""
    parent_transform = (
        np.eye(4)
        if parent_pose is None
        else poseul2mat4d(
            position=parent_pose[:3],
            euler=parent_pose[3:],
        )
    )
    link_transform = poseul2mat4d(
        position=child_pose[:3],
        euler=child_pose[3:],
    )
    local_transform = np.linalg.inv(parent_transform) @ link_transform
    return local_transform[:3, -1], Rotation.from_matrix(
        local_transform[:3, :3]
    ).as_euler('xyz')


def grid_material(mjcf_model):
    """Get grid texture"""
    texture = mjcf_model.asset.find(
        namespace='texture',
        identifier='texture_grid',
    )
    if texture is None:
        texture = mjcf_model.asset.add(  # Add grid texture to assets
            'texture',
            name='texture_grid',
            type='2d',
            builtin='checker',
            rgb1=[0.1, 0.2, 0.3],
            rgb2=[0.2, 0.3, 0.4],
            width=300,
            height=300,
            mark='edge',
            markrgb=[0.2, 0.3, 0.4],
        )
    material = mjcf_model.asset.find(
        namespace='material',
        identifier='material_grid',
    )
    if material is None:
        material = mjcf_model.asset.add(  # Add grid material to assets
            'material',
            name='material_grid',
            texture='texture_grid',
            texrepeat=[1, 1],
            texuniform=True,
            reflectance=0.2,
        )
    return material, texture


def mjc_add_link(mjcf_model, mjcf_map, sdf_link, **kwargs):
    """Add link to world"""

    sdf_parent = kwargs.pop('sdf_parent', None)
    mjc_parent = kwargs.pop('mjc_parent', None)
    sdf_joint = kwargs.pop('sdf_joint', None)
    directory = kwargs.pop('directory', '')
    free = kwargs.pop('free', False)
    self_collisions = kwargs.pop('self_collisions', False)
    concave = kwargs.pop('concave', False)
    overwrite = kwargs.pop('overwrite', True)
    solref = kwargs.pop('solref', None)  # [0.02, 1]
    solimp = kwargs.pop('solimp', None)  # [0.9, 0.95, 0.001, 0.5, 2]
    friction = kwargs.pop('friction', [1.0, 0, 0])
    frictionloss = kwargs.pop('frictionloss', 1e-5)
    damping = kwargs.pop('damping', 0)
    use_site = kwargs.pop('use_site', False)
    units = kwargs.pop('units', SimulationUnitScaling())
    assert not kwargs, kwargs

    # Links (bodies)
    if mjc_parent is None or sdf_parent is None:
        mjc_parent = mjcf_model.worldbody
    link_local_pos, link_local_euler = get_local_transform(
        parent_pose=None if sdf_parent is None else sdf_parent.pose,
        child_pose=sdf_link.pose,
    )
    body = mjc_parent.add(
        'body',
        name=sdf_link.name,
        pos=[pos*units.meters for pos in link_local_pos],
        quat=euler2mjcquat(link_local_euler),
    )
    mjcf_map['links'][sdf_link.name] = body

    # Site
    if use_site:
        site = body.add(
            'site',
            type='box',
            name=f'site_{sdf_link.name}',
            group=1,
            pos=[0, 0, 0],
            quat=[1, 0, 0, 0],
            size=[1e-2*units.meters]*3,
        )
        mjcf_map['sites'][site.name] = site

    # joints
    joint = None
    if free:  # Freejoint
        joint = body.add('freejoint', name=f'root_{sdf_link.name}')
    elif sdf_joint is not None:
        if sdf_joint.type in ('revolute', 'continuous'):
            joint = body.add(
                'joint',
                name=sdf_joint.name,
                axis=sdf_joint.axis.xyz,
                pos=[pos*units.meters for pos in sdf_joint.pose[:3]],
                # euler=sdf_joint.pose[3:],  # Euler not supported in joint
                type='hinge',
                damping=damping*units.damping,
                frictionloss=frictionloss,
            )
            mjcf_map['joints'][sdf_joint.name] = joint

    # Visual and collisions (geoms)
    for element in sdf_link.collisions + sdf_link.visuals:

        geom = None

        # Include in mjcf
        visual_kwargs = {}
        collision_kwargs = {}
        geom_kwargs = {
            'name': element.name,
            'pos': [pos*units.meters for pos in element.pose[:3]],
            'quat': euler2mjcquat(element.pose[3:]),
        }
        if isinstance(element, Visual):
            if element.color is not None:
                mjcf_model.asset.add(  # Add material to assets
                    'material',
                    name=f'material_{element.name}',
                    rgba=element.color,
                )
                visual_kwargs['material'] = f'material_{element.name}'
            visual_kwargs['conaffinity'] = 0  # No self-collisions
            visual_kwargs['contype'] = 0  # No world collisions
            visual_kwargs['group'] = 1
        elif isinstance(element, Collision):
            collision_kwargs['friction'] = friction
            collision_kwargs['margin'] = 0
            collision_kwargs['contype'] = 1  # World collisions
            collision_kwargs['conaffinity'] = self_collisions
            collision_kwargs['condim'] = 3
            collision_kwargs['group'] = 2
            if solref is not None:
                if all(sol < 0 for sol in solref):
                    solref[0] *= units.newtons/units.meters
                    solref[1] *= units.newtons/units.velocity
                else:
                    solref[0] *= units.seconds
                collision_kwargs['solref'] = solref
            if solimp is not None:
                collision_kwargs['solimp'] = solimp

        # Mesh
        if isinstance(element.geometry, Mesh):

            # Mesh path
            mesh_path = os.path.join(directory, element.geometry.uri)
            assert os.path.isfile(mesh_path)

            # Convert to STL if mesh in other format
            path, extension = os.path.splitext(mesh_path)
            if extension != '.stl':
                stl_path = f'{path}.stl'
                if overwrite or not os.path.isfile(stl_path):
                    mesh = tri.load_mesh(mesh_path)
                    if isinstance(mesh, tri.Scene):
                        mesh = tri.util.concatenate(
                            tuple(tri.Trimesh(
                                vertices=g.vertices,
                                faces=g.faces,
                            )
                            for g in mesh.geometry.values())
                        )
                    if not mesh.convex_hull.vertices.any():
                        continue
                    mesh.export(stl_path)
                mesh_path = stl_path

            # Convexify
            mesh = tri.load_mesh(mesh_path)
            if (
                    isinstance(element, Collision)
                    and concave
                    and not tri.convex.is_convex(mesh)
            ):
                pylog.info('Convexifying %s', mesh_path)
                meshes = tri.interfaces.vhacd.convex_decomposition(
                    mesh,
                    # Original parameters
                    # resolution=100000,
                    # concavity=0.001,
                    # planeDownsampling=4,
                    # convexhullDownsampling=4,
                    # alpha=0.05,
                    # beta=0.05,
                    # maxhulls=1024,
                    # pca=0,
                    # mode=0,
                    # maxNumVerticesPerCH=64,
                    # minVolumePerCH=0.0001,
                    # convexhullApproximation=1,
                    # oclAcceleration=1,
                    # oclPlatformId=0,
                    # oclDevideID=0,
                    # Parameters
                    resolution=int(1e5),
                    concavity=1e-8,
                    planeDownsampling=1,
                    convexhullDownsampling=1,
                    alpha=0.05,
                    beta=0.05,
                    gamma=0.00125,
                    delta=0.05,
                    maxhulls=int(1e4),
                    pca=0,
                    mode=1,
                    maxNumVerticesPerCH=int(1e4),
                    minVolumePerCH=1e-10,
                    convexhullApproximation=1,
                    oclAcceleration=1,
                )
                pylog.info('Convex decomposiion complete')
                original_path, extension = os.path.splitext(mesh_path)
                name = geom_kwargs['name']
                for mesh_i, mesh in enumerate(meshes):
                    path = f'{original_path}_convex_{mesh_i}{extension}'
                    pylog.info('Exporting and loading %s', path)
                    mesh.export(path)
                    mjcf_model.asset.add(  # Add mesh to assets
                        'mesh',
                        name=f'mesh_{element.name}_convex_{mesh_i}',
                        file=path,
                        scale=[s*units.meters for s in element.geometry.scale],
                    )
                    geom_kwargs['name'] = f'{name}_convex_{mesh_i}'
                    _geom = body.add(
                        'geom',
                        type='mesh',
                        mesh=f'mesh_{element.name}_convex_{mesh_i}',
                        **geom_kwargs,
                        **visual_kwargs,
                        **collision_kwargs,
                    )
                    pylog.info('loaded %s', path)
                    if not mesh_i:
                        geom = _geom
            else:
                # Add mesh asset
                mjcf_model.asset.add(  # Add mesh to assets
                    'mesh',
                    name=f'mesh_{element.name}',
                    file=mesh_path,
                    scale=[s*units.meters for s in element.geometry.scale],
                )

                geom = body.add(
                    'geom',
                    type='mesh',
                    mesh=f'mesh_{element.name}',
                    **geom_kwargs,
                    **visual_kwargs,
                    **collision_kwargs,
                )

        # Box
        elif isinstance(element.geometry, Box):

            geom = body.add(
                'geom',
                type='box',
                size=[0.5*s*units.meters for s in element.geometry.size],
                **geom_kwargs,
                **visual_kwargs,
                **collision_kwargs,
            )

        # Cylinder
        elif isinstance(element.geometry, Cylinder):

            geom = body.add(
                'geom',
                type='cylinder',
                size=[  # Radius, length, unused
                    element.geometry.radius*units.meters,
                    0.5*element.geometry.length*units.meters,
                    element.geometry.radius*units.meters,
                ],
                **geom_kwargs,
                **visual_kwargs,
                **collision_kwargs,
            )

        # Capsule
        elif isinstance(element.geometry, Capsule):

            geom = body.add(
                'geom',
                type='capsule',
                size=[  # Radius, length, unused
                    element.geometry.radius*units.meters,
                    max(
                        element.geometry.length-2*element.geometry.radius,
                        element.geometry.radius,
                    )*units.meters,
                    element.geometry.radius*units.meters,
                ],
                **geom_kwargs,
                **visual_kwargs,
                **collision_kwargs,
            )

        # Sphere
        elif isinstance(element.geometry, Sphere):

            geom = body.add(
                'geom',
                type='sphere',
                size=[element.geometry.radius*units.meters]*3,
                **geom_kwargs,
                **visual_kwargs,
                **collision_kwargs,
            )

        elif isinstance(element.geometry, Plane):

            material, _ = grid_material(mjcf_model)
            geom = body.add(
                'geom',
                type='plane',
                size=element.geometry.size,
                material=material.name,
                **geom_kwargs,
                **visual_kwargs,
                **collision_kwargs,
            )

        else:
            raise NotImplementedError(f'{type(element.geometry)} not supported')

        if geom is not None:
            if isinstance(element, Visual):
                mjcf_map['visuals'][element.name] = geom
            elif isinstance(element, Collision):
                mjcf_map['collisions'][element.name] = geom

    # Inertial
    inertial = sdf_link.inertial
    if inertial is not None:
        inertia_mat = np.diag([
            max(MIN_INERTIA, inertial.inertias[0]),
            max(MIN_INERTIA, inertial.inertias[3]),
            max(MIN_INERTIA, inertial.inertias[5]),
        ])
        inertia_mat[0][1] = inertial.inertias[1]
        inertia_mat[1][0] = inertial.inertias[1]
        inertia_mat[0][2] = inertial.inertias[2]
        inertia_mat[2][0] = inertial.inertias[2]
        inertia_mat[1][2] = inertial.inertias[4]
        inertia_mat[2][1] = inertial.inertias[4]
        mat = euler2mat(inertial.pose[3:])
        inertia_mat = mat @ inertia_mat @ mat.T
        eigvals = np.linalg.eigvals(inertia_mat)
        assert (eigvals > 0).all(), f'Eigen values <= 0: {eigvals}\n{inertia_mat}'
        body.add(
            'inertial',
            pos=[pos*units.meters for pos in inertial.pose[:3]],
            # quat=euler2mjcquat(inertial.pose[3:]),  # Not working??
            mass=inertial.mass*units.kilograms,
            # diaginertia=np.clip([
            #     inertial.inertias[0],
            #     inertial.inertias[3],
            #     inertial.inertias[5]
            # ], 0, np.inf).tolist(),
            fullinertia=[
                # max(MIN_INERTIA, inertial.inertias[0])*units.inertia,
                # max(MIN_INERTIA, inertial.inertias[3])*units.inertia,
                # max(MIN_INERTIA, inertial.inertias[5])*units.inertia,
                # inertial.inertias[1]*units.inertia,
                # inertial.inertias[2]*units.inertia,
                # inertial.inertias[4]*units.inertia,
                # Since quat does not seem to have any effect, the tensor is
                # rotated beforehand to set the right inertial orientation
                max(MIN_INERTIA, inertia_mat[0][0])*units.inertia,
                max(MIN_INERTIA, inertia_mat[1][1])*units.inertia,
                max(MIN_INERTIA, inertia_mat[2][2])*units.inertia,
                inertia_mat[0][1]*units.inertia,
                inertia_mat[0][2]*units.inertia,
                inertia_mat[1][2]*units.inertia,
            ],
        )
    elif not free:
        body.add(
            'inertial',
            pos=[0, 0, 0],
            quat=[1, 0, 0, 0],
            mass=1,
            diaginertia=[1, 1, 1],
        )

    return body, joint


def add_link_recursive(mjcf_model, mjcf_map, sdf, **kwargs):
    """Add link recursive"""

    # Handle kwargs
    sdf_link = kwargs.pop('sdf_link', None)
    sdf_parent = kwargs.pop('sdf_parent', None)
    sdf_joint = kwargs.pop('sdf_joint', None)
    free = kwargs.pop('free', False)

    # Add link
    mjc_add_link(
        mjcf_model=mjcf_model,
        mjcf_map=mjcf_map,
        sdf_link=sdf_link,
        sdf_parent=sdf_parent,
        sdf_joint=sdf_joint,
        directory=sdf.directory,
        free=free,
        mjc_parent=(
            mjcf_map['links'].get(sdf_parent.name)
            if sdf_parent is not None
            else None
        ),
        **kwargs,
    )

    # Add children
    for child in sdf.get_children(link=sdf_link):
        add_link_recursive(
            mjcf_model=mjcf_model,
            mjcf_map=mjcf_map,
            sdf=sdf,
            sdf_link=child,
            sdf_parent=sdf_link,
            sdf_joint=sdf.get_parent_joint(link=child),
            **kwargs
        )


def sdf2mjcf(sdf, **kwargs):
    """Export to MJCF string"""

    mjcf_model = kwargs.pop('mjcf_model', None)
    fixed_base = kwargs.pop('fixed_base', False)
    concave = kwargs.pop('concave', False)
    use_site = kwargs.pop('use_site', False)
    use_sensors = kwargs.pop('use_sensors', False)
    use_actuators = kwargs.pop('use_actuators', False)
    # Position
    act_pos_gain = kwargs.pop('act_pos_gain', 1)
    act_pos_ctrllimited = kwargs.pop('act_pos_ctrllimited', False)
    act_pos_ctrlrange = kwargs.pop('act_pos_ctrlrange', [0, 0])
    act_pos_forcelimited = kwargs.pop('act_pos_forcelimited', False)
    act_pos_forcerange = kwargs.pop('act_pos_forcerange', [0, 0])
    # Velocity
    act_vel_gain = kwargs.pop('act_vel_gain', 1)
    act_vel_ctrllimited = kwargs.pop('act_vel_ctrllimited', False)
    act_vel_ctrlrange = kwargs.pop('act_vel_ctrlrange', [0, 0])
    act_vel_forcelimited = kwargs.pop('act_vel_forcelimited', False)
    act_vel_forcerange = kwargs.pop('act_vel_forcerange', [0, 0])
    # Animat options
    animat_options = kwargs.pop('animat_options', None)
    simulation_options = kwargs.pop('simulation_options', None)
    units = kwargs.pop('units', (
        simulation_options.units
        if simulation_options is not None
        else SimulationUnitScaling()
    ))

    if mjcf_model is None:
        mjcf_model = mjcf.RootElement()

    # Base link
    roots = sdf.get_base_links()

    # Elements
    mjcf_map = {
        element: {}
        for element in [
                'links', 'joints',
                'sites', 'visuals', 'collisions',
                'actuators',
        ]
    }

    # Add trees from roots
    for root in roots:
        add_link_recursive(
            mjcf_model=mjcf_model,
            mjcf_map=mjcf_map,
            sdf=sdf,
            sdf_link=root,
            free=not fixed_base,
            use_site=use_site,
            self_collisions=fixed_base,
            concave=concave,
            units=units,
            **kwargs
        )

    # Actuators
    if use_actuators:
        joints_names = (
            [joint.joint_name for joint in animat_options.control.joints]
            if animat_options is not None
            else mjcf_map['joints']
        )
        for joint_name in joints_names:
            if mjcf_model.find('joint', joint_name).type != 'hinge':
                continue
            # mjcf_model.actuator.add(
            #     'general',
            #     name=f'act_pos_{joint_name}',
            #     joint=joint_name,
            # )
            name = f'actuator_position_{joint_name}'
            mjcf_map['actuators'][name] = mjcf_model.actuator.add(
                'position',
                name=name,
                joint=joint_name,
                kp=act_pos_gain*units.torques,
                ctrllimited=act_pos_ctrllimited,
                ctrlrange=act_pos_ctrlrange,
                forcelimited=act_pos_forcelimited,
                forcerange=[val*units.torques for val in act_pos_forcerange],
            )
            name = f'actuator_velocity_{joint_name}'
            mjcf_map['actuators'][name] = mjcf_model.actuator.add(
                'velocity',
                name=name,
                joint=joint_name,
                kv=act_vel_gain*units.torques/units.angular_velocity,
                ctrllimited=act_vel_ctrllimited,
                ctrlrange=[val*units.angular_velocity for val in act_vel_ctrlrange],
                forcelimited=act_vel_forcelimited,
                forcerange=[val*units.torques for val in act_vel_forcerange],
            )
            name = f'actuator_torque_{joint_name}'
            mjcf_map['actuators'][name] = mjcf_model.actuator.add(
                'motor',
                name=name,
                joint=joint_name,
            )
        assert mjcf_map['actuators'], mjcf_map['actuators']

    # Sensors
    if use_sensors:
        for link_name in mjcf_map['links']:
            for link_sensor in (
                    'framepos', 'framequat',
                    'framelinvel', 'frameangvel',
            ):
                mjcf_model.sensor.add(
                    link_sensor,
                    name=f'{link_sensor}_{link_name}',
                    objname=link_name,
                    objtype='body',
                )
            if use_site:
                for link_sensor in ('touch',):  # 'velocimeter', 'gyro',
                    mjcf_model.sensor.add(
                        link_sensor,
                        name=f'{link_sensor}_{link_name}',
                        site=f'site_{link_name}',
                    )
        for joint_name in mjcf_map['joints']:
            for joint_sensor in ('jointpos', 'jointvel'):
                mjcf_model.sensor.add(
                    joint_sensor,
                    name=f'{joint_sensor}_{joint_name}',
                    joint=joint_name,
                )
        for actuator_name, actuator in mjcf_map['actuators'].items():
            mjcf_model.sensor.add(
                'actuatorfrc',
                name=f'actuatorfrc_{actuator.tag}_{actuator.joint}',
                actuator=actuator_name,
            )

    return mjcf_model


def mjcf2str(mjcf_model):
    """Export to MJCF string"""
    xml_str = ET.tostring(
        mjcf_model.to_xml(),
        encoding='utf8',
        method='xml'
    ).decode('utf8')
    dom = xml.dom.minidom.parseString(xml_str)
    return dom.toprettyxml(indent=2*' ')


def night_sky(mjcf_model):
    """Night sky"""
    mjcf_model.asset.add(  # Add night sky texture to assets
        'texture',
        name='skybox',
        type='skybox',
        builtin='gradient',
        rgb1=[0.4, 0.6, 0.8],
        rgb2=[0.0, 0.0, 0.0],
        width=800,
        height=800,
        mark='random',
        markrgb=[1.0, 1.0, 1.0],
    )


def add_plane(mjcf_model):
    """Add plane"""
    material, texture = grid_material(mjcf_model)
    geometry = mjcf_model.worldbody.add(  # Add floor
        'geom',
        type='plane',
        name='floor',
        pos=[0, 0, 0],
        size=[10, 10, 0.1],
        material=material.name,
        friction=[1e0, 0, 0],
    )
    return geometry, material, texture


def add_particles(mjcf_model):
    """Add particles"""
    composite = mjcf_model.worldbody.add(
        'composite',
        type='particle',
        prefix='s',
        count=[10, 10, 10],
        spacing=0.07,
        offset=[0, 0, 1],
    )
    composite.geom.size = [0.02]
    composite.geom.rgba = [0.8, 0.2, 0.1, 1.0]


def add_lights(mjcf_model, link, meters=1):
    """Add lights"""
    mjcf_model.visual.quality.shadowsize = 32*1024
    for i, pos in enumerate([[-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]]):
        link.add(
            'light',
            name=f'light_{link.name}_{i}',
            mode='trackcom',
            # target='targetbody',
            active=str(True).lower(),
            pos=[val*meters for val in pos],
            castshadow=str(not i).lower(),
            directional=str(False).lower(),
            dir=[0, 0, -1.0],
            attenuation=[2.0, 0.0, 0.0],
            cutoff=120,
            exponent=10,
            ambient=[0.0, 0.0, 0.0],
            diffuse=[0.7, 0.7, 0.7],
            specular=[0.3, 0.3, 0.3],
        )


def add_cameras(mjcf_model, link, meters=1, dist=0.3):
    """Add cameras"""
    mjcf_model.visual.quality.shadowsize = 32*1024
    for i, (mode, pose) in enumerate([
            ['trackcom', [0.0, 0.0, dist, 0.0, 0.0, 0.0]],
            ['trackcom', [0.0, -dist, 0.2*dist, 0.4*np.pi, 0.0, 0.0]],
            ['trackcom', [dist, 0.0, 0.2*dist, 0.0, 0.4*np.pi, 0.5*np.pi]],
            ['targetbodycom', [dist, 0.0, 0.2*dist, 0.0, 0.4*np.pi, 0.5*np.pi]],
    ]):
        link.add(
            'camera',
            name=f'camera_{link.name}_{i}',
            mode=mode,
            pos=[pos*meters for pos in pose[:3]],
            quat=euler2mjcquat(pose[3:]),
            # target='',
            fovy=45,
            ipd=0.068,
        )


def setup_mjcf_xml(sdf_path_animat, sdf_path_arena, **kwargs):
    """Setup MJCF XML"""

    animat_options = kwargs.pop('animat_options', None)
    simulation_options = kwargs.pop('simulation_options', None)
    units = kwargs.pop('units', (
        simulation_options.units
        if simulation_options is not None
        else SimulationUnitScaling()
    ))
    timestep = kwargs.pop(
        'timestep',
        simulation_options.timestep
        if simulation_options is not None
        else 1e-3,
    )

    # Arena
    mjcf_model = sdf2mjcf(
        sdf=ModelSDF.read(filename=os.path.expandvars(sdf_path_arena))[0],
        # mjcf_model=mjcf_model,
        fixed_base=True,
        concave=False,
        simulation_options=simulation_options,
    )
    # add_plane(mjcf_model)

    # Animat
    sdf_animat = ModelSDF.read(filename=os.path.expandvars(sdf_path_animat))[0]
    mjcf_model = sdf2mjcf(
        sdf=sdf_animat,
        mjcf_model=mjcf_model,
        use_sensors=True,
        use_actuators=True,
        animat_options=animat_options,
        simulation_options=simulation_options,
        # Links
        friction=[1.0, 1e-3, 1e-4],
        # Joints
        damping=1e-4,
        act_pos_gain=5e-3,
        act_vel_gain=1e-4,
    )

    # Spawn options
    sdf_base_link = sdf_animat.get_base_link()
    base_link = mjcf_model.worldbody.body[-1]
    assert base_link.name == sdf_base_link.name
    base_link.pos = kwargs.pop('spawn_position', [0, 0, 0])
    base_link.quat = euler2mjcquat(kwargs.pop('spawn_rotation', [0, 0, 0]))

    # Compiler
    mjcf_model.compiler.angle = 'radian'
    mjcf_model.compiler.eulerseq = 'xyz'
    mjcf_model.compiler.boundmass = MIN_MASS
    mjcf_model.compiler.boundinertia = MIN_INERTIA
    mjcf_model.compiler.balanceinertia = False
    mjcf_model.compiler.inertiafromgeom = False
    mjcf_model.compiler.convexhull = False
    mjcf_model.compiler.discardvisual = kwargs.pop('discardvisual', False)

    # Simulation options
    mjcf_model.size.njmax = 2**12
    mjcf_model.size.nconmax = 2**12
    mjcf_model.option.gravity = kwargs.pop('gravity', [0, 0, -9.81])
    mjcf_model.option.timestep = timestep
    mjcf_model.option.iterations = kwargs.pop('solver_iterations', 100)
    mjcf_model.option.solver = 'Newton'  # PGS, CG, Newton
    mjcf_model.option.integrator = 'Euler'  # Euler, RK4
    mjcf_model.option.mpr_iterations = kwargs.pop('mpr_iterations', 50)

    # Animat options
    if animat_options is not None:
        for link in animat_options.morphology.links:
            pass
        for joint in animat_options.morphology.joints:
            pass
        base_link.pos = [pos*units.meters for pos in animat_options.spawn.position]
        base_link.quat = euler2mjcquat(animat_options.spawn.orientation)

    if simulation_options is not None:
        mjcf_model.option.gravity = [
            garvity*units.acceleration
            for garvity in simulation_options.gravity
        ]
        mjcf_model.option.timestep = timestep
        mjcf_model.option.iterations = simulation_options.n_solver_iters

    # Add particles
    if kwargs.pop('use_particles', False):
        add_particles(mjcf_model)

    # Light and shadows
    add_lights(mjcf_model=mjcf_model, link=base_link, meters=units.meters)

    # Add cameras
    add_cameras(mjcf_model=mjcf_model, link=base_link, meters=units.meters)

    # Night sky
    night_sky(mjcf_model)

    # XML string
    mjcf_xml_str = mjcf2str(mjcf_model=mjcf_model)
    pylog.info(mjcf_xml_str)
    if kwargs.pop('save_mjcf', False):
        with open('simulation_mjcf.xml', 'w+') as xml_file:
            xml_file.write(mjcf_xml_str)

    assert not kwargs, kwargs
    return mjcf_model, base_link
