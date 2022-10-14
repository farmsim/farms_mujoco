"""MJCF handling"""

import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
from typing import Dict

import numpy as np
import trimesh as tri
from imageio import imread
from scipy.spatial.transform import Rotation

from dm_control import mjcf

from farms_core import pylog
from farms_core.units import SimulationUnitScaling
from farms_core.simulation.options import SimulationOptions
from farms_core.array.types import (
    NDARRAY_3,
    NDARRAY_4,
    NDARRAY_6,
    NDARRAY_33,
    NDARRAY_44,
)
from farms_core.sensors.sensor_convention import sc
from farms_core.io.yaml import read_yaml
from farms_core.io.sdf import (
    ModelSDF, Link, Mesh, Visual, Collision,
    Box, Cylinder, Capsule, Sphere, Plane, Heightmap,
)


MIN_MASS = 1e-6
MIN_INERTIA = 1e-12


def quat2mjcquat(quat: NDARRAY_6) -> NDARRAY_4:
    """Quaternion to MuJoCo quaternion"""
    quat_type = np.array if isinstance(quat, np.ndarray) else type(quat)
    quat = np.array(quat)[[3, 0, 1, 2]]
    return quat_type(quat)


def euler2mjcquat(euler: NDARRAY_3) -> NDARRAY_4:
    """Euler to MuJoCo quaternion"""
    return quat2mjcquat(Rotation.from_euler(
        angles=euler,
        seq='xyz',
    ).as_quat())


def euler2mat(euler: NDARRAY_3) -> NDARRAY_33:
    """Euler to 3D matrix"""
    return Rotation.from_euler(
        angles=euler,
        seq='xyz',
    ).as_matrix()


def poseul2mat4d(
        position: NDARRAY_3,
        euler: NDARRAY_3,
) -> NDARRAY_44:
    """4D transform"""
    transform = np.eye(4)
    transform[:3, -1] = position
    transform[:3, :3] = euler2mat(euler)
    return transform


def get_local_transform(
        parent_pose: NDARRAY_6,
        child_pose: NDARRAY_6,
):
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


def grid_material(mjcf_model: mjcf.Element) -> (mjcf.Element, mjcf.Element):
    """Get grid texture"""
    texture = mjcf_model.asset.find(
        namespace='texture',
        identifier='texture_grid',
    )
    if texture is None:
        texture = mjcf_model.asset.add(
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
        material = mjcf_model.asset.add(
            'material',
            name='material_grid',
            texture='texture_grid',
            texrepeat=[1, 1],
            texuniform=True,
            reflectance=0.2,
        )
    return material, texture


def mjc_add_link(
        mjcf_model: mjcf.RootElement,
        mjcf_map: Dict,
        sdf_link: Link,
        **kwargs,
) -> (mjcf.Element, mjcf.Element):
    """Add link to world"""

    sdf_parent = kwargs.pop('sdf_parent', None)
    mjc_parent = kwargs.pop('mjc_parent', None)
    sdf_joint = kwargs.pop('sdf_joint', None)
    directory = kwargs.pop('directory', '')
    free = kwargs.pop('free', False)
    all_collisions = kwargs.pop('all_collisions', False)
    concave = kwargs.pop('concave', False)
    overwrite = kwargs.pop('overwrite', False)
    solref = kwargs.pop('solref', None)
    solimp = kwargs.pop('solimp', None)
    friction = kwargs.pop('friction', [0, 0, 0])
    use_site = kwargs.pop('use_site', False)
    units = kwargs.pop('units', SimulationUnitScaling())
    assert not kwargs, kwargs

    # Links (bodies)
    link_name = sdf_link.name
    if mjc_parent is None or sdf_parent is None:
        mjc_parent = mjcf_model.worldbody
    link_local_pos, link_local_euler = get_local_transform(
        parent_pose=None if sdf_parent is None else sdf_parent.pose,
        child_pose=sdf_link.pose,
    )
    body = mjcf_model.worldbody if link_name == 'world' else mjc_parent.add(
        'body',
        name=link_name,
        pos=[pos*units.meters for pos in link_local_pos],
        quat=euler2mjcquat(link_local_euler),
    )
    mjcf_map['links'][link_name] = body

    # joints
    joint = None
    if isinstance(sdf_link, ModelSDF):
        if free:  # Freejoint
            joint = body.add('freejoint', name=f'root_{link_name}')
        return body, joint
    if not free and sdf_joint is not None:
        if sdf_joint.type in ('revolute', 'continuous'):
            joint = body.add(
                'joint',
                name=sdf_joint.name,
                axis=sdf_joint.axis.xyz,
                pos=[pos*units.meters for pos in sdf_joint.pose[:3]],
                # euler=sdf_joint.pose[3:],  # Euler not supported in joint
                type='hinge',
                damping=0,
                stiffness=0,
                springref=0,
                frictionloss=0,
                limited=True if sdf_joint.axis.limits else False,
                range=sdf_joint.axis.limits[:2] if sdf_joint.axis.limits else [-1.0, 1.0]
            )
            mjcf_map['joints'][sdf_joint.name] = joint
        elif sdf_joint.type in ('prismatic'):
            joint = body.add(
                'joint',
                name=sdf_joint.name,
                axis=sdf_joint.axis.xyz,
                pos=[pos*units.meters for pos in sdf_joint.pose[:3]],
                # euler=sdf_joint.pose[3:],  # Euler not supported in joint
                type='slide',
                damping=0,
                stiffness=0,
                springref=0,
                frictionloss=0,
                limited=True if sdf_joint.axis.limits else False,
                range=sdf_joint.axis.limits[:2] if sdf_joint.axis.limits else [-1.0, 1.0]
            )
            mjcf_map['joints'][sdf_joint.name] = joint

    # Site
    if use_site:
        site = body.add(
            'site',
            type='box',
            name=f'site_{link_name}',
            group=1,
            pos=[0, 0, 0],
            quat=[1, 0, 0, 0],
            size=[1e-2*units.meters]*3,
        )
        mjcf_map['sites'][site.name] = site

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
            collision_kwargs['conaffinity'] = all_collisions
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
                        mesh = tri.util.concatenate(tuple(
                            tri.Trimesh(vertices=g.vertices, faces=g.faces)
                            for g in mesh.geometry.values()
                        ))
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
                    concavity=1e-6,
                    planeDownsampling=4,
                    convexhullDownsampling=4,
                    alpha=0.05,
                    beta=0.05,
                    gamma=0.00125,
                    delta=0.05,
                    maxhulls=1024,
                    pca=0,
                    mode=0,
                    maxNumVerticesPerCH=1024,
                    minVolumePerCH=1e-6,
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
                    mjcf_model.asset.add(
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
                mjcf_model.asset.add(
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
                size=[
                    element.geometry.radius*units.meters,  # Radius
                    0.5*element.geometry.length*units.meters,  # Half-length
                    0,  # Unused
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

        # Plane
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

        # Heightmap
        elif isinstance(element.geometry, Heightmap):

            material, _ = grid_material(mjcf_model)
            path = os.path.join(directory, element.geometry.uri)
            assert os.path.isfile(path), path
            img = imread(path)  # Read PNG image
            img = img[:, :, 0] if img.ndim == 3 else img[:, :]  # RGB vs Grey
            vmin, vmax = (np.iinfo(img.dtype).min, np.iinfo(img.dtype).max)
            img = (img - vmin)/(vmax-vmin)  # Normalize
            img = np.flip(img, axis=0)  # Cartesian coordinates
            mjcf_map['hfield'] = {
                'data': img,
                'asset': mjcf_model.asset.add(
                    'hfield',
                    name=element.name,
                    nrow=img.shape[0],
                    ncol=img.shape[1],
                    size=(
                        0.5*element.geometry.size[0]*units.meters,
                        0.5*element.geometry.size[1]*units.meters,
                        0.5*element.geometry.size[2]*units.meters,
                        0.5*element.geometry.size[2]*units.meters,
                    )
                )
            }
            # geom_kwargs['pos'] += mjcf_map['hfield']['asset'].size[:3]
            collision_kwargs['group'] = 3
            geom = body.add(
                'geom',
                type='hfield',
                size=element.geometry.size,
                material=material.name,
                hfield=mjcf_map['hfield']['asset'].name,
                **geom_kwargs,
                **visual_kwargs,
                **collision_kwargs,
            )

        else:
            raise NotImplementedError(
                f'{type(element.geometry)} not supported'
            )

        if geom is not None:
            if isinstance(element, Visual):
                mjcf_map['visuals'][element.name] = geom
            elif isinstance(element, Collision):
                mjcf_map['collisions'][element.name] = geom

    # Inertial
    inertial = None if link_name == 'world' else sdf_link.inertial
    if inertial is not None:

        # Extract and validate inertia
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
        eigvals = np.linalg.eigvals(inertia_mat)
        assert (eigvals > 0).all(), (
            f'Eigen values <= 0 for link {link_name}'
            f'\nEigenvalues: {eigvals}'
            f'\nInertia:\n{inertia_mat}'
        )

        # Rotate inertia tensor
        rot_mat = euler2mat(inertial.pose[3:])
        inertia_mat = rot_mat @ inertia_mat @ rot_mat.T

        # Add inertial
        body.add(
            'inertial',
            pos=[pos*units.meters for pos in inertial.pose[:3]],
            # quat=euler2mjcquat(inertial.pose[3:]),  # Not working in MuJoCo?
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

    elif body is not mjcf_model.worldbody:
        body.add(
            'inertial',
            pos=[0, 0, 0],
            quat=[1, 0, 0, 0],
            mass=1e-10,
            diaginertia=[1e-12]*3,
        )

    return body, joint


def add_link_recursive(
        mjcf_model: mjcf.RootElement,
        mjcf_map: Dict,
        sdf: str,
        **kwargs,
):
    """Add link recursive"""

    # Handle kwargs
    sdf_link = kwargs.pop('sdf_link')
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


def sdf2mjcf(
        sdf: str,
        **kwargs,
) -> (mjcf.RootElement, Dict):
    """Export to MJCF string"""

    mjcf_model = kwargs.pop('mjcf_model', None)
    model_name = kwargs.pop('model_name', None)
    fixed_base = kwargs.pop('fixed_base', False)
    concave = kwargs.pop('concave', False)
    use_site = kwargs.pop('use_site', False)
    use_sensors = kwargs.pop('use_sensors', False)
    use_link_sensors = kwargs.pop('use_link_sensors', True)
    use_link_vel_sensors = kwargs.pop('use_link_vel_sensors', True)
    use_joint_sensors = kwargs.pop('use_joint_sensors', True)
    use_actuator_sensors = kwargs.pop('use_actuator_sensors', True)
    use_muscle_sensors = kwargs.pop('use_muscle_sensors', True)
    use_actuators = kwargs.pop('use_actuators', False)
    use_muscles = kwargs.pop('use_muscles', True)
    solref = kwargs.get('solref', None)

    # Position
    act_pos_ctrllimited = kwargs.pop('act_pos_ctrllimited', False)
    act_pos_ctrlrange = kwargs.pop('act_pos_ctrlrange', [-1e6, 1e6])
    act_pos_forcelimited = kwargs.pop('act_pos_forcelimited', False)
    act_pos_forcerange = kwargs.pop('act_pos_forcerange', [-1e6, 1e6])

    # Velocity
    act_vel_ctrllimited = kwargs.pop('act_vel_ctrllimited', False)
    act_vel_ctrlrange = kwargs.pop('act_vel_ctrlrange', [-1e6, 1e6])
    act_vel_forcelimited = kwargs.pop('act_vel_forcelimited', False)
    act_vel_forcerange = kwargs.pop('act_vel_forcerange', [-1e6, 1e6])

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

    # Name
    if model_name:
        mjcf_model.model = model_name

    # Base link
    roots = sdf.get_base_links()

    # Elements
    mjcf_map = {
        element: {}
        for element in [
                'links', 'joints',
                'sites', 'visuals', 'collisions',
                'actuators', 'tendons', 'muscles'
        ]
    }

    # Add model root link
    mjc_add_link(
        mjcf_model=mjcf_model,
        mjcf_map=mjcf_map,
        sdf_link=sdf,
        sdf_parent=None,
        sdf_joint=None,
        directory=sdf.directory,
        free=not fixed_base,
        mjc_parent=None,
        **kwargs,
    )

    # Add trees from roots
    for root in roots:
        add_link_recursive(
            mjcf_model=mjcf_model,
            mjcf_map=mjcf_map,
            sdf=sdf,
            sdf_link=root,
            sdf_parent=sdf,
            free=not fixed_base,
            use_site=use_site,
            concave=concave,
            units=units,
            **kwargs
        )

    # Actuators
    if use_actuators:
        joints_names = (
            animat_options.control.joints_names()
            if animat_options is not None
            else mjcf_map['joints']
        )
        if animat_options is not None:
            motors_ctrl = {
                motor.joint_name: motor
                for motor in animat_options.control.motors
            }
        for joint_name in joints_names:
            joint = mjcf_model.find('joint', joint_name)
            assert joint, f'Joint {joint_name} not found'
            if joint.type != 'hinge':
                # continue
                raise Exception(
                    f'Actuator is not of type hinge but of type {joint.type}'
                )
            # mjcf_model.actuator.add(
            #     'general',
            #     name=f'act_pos_{joint_name}',
            #     joint=joint_name,
            # )
            name_pos = f'actuator_position_{joint_name}'
            mjcf_map['actuators'][name_pos] = mjcf_model.actuator.add(
                'position',
                name=name_pos,
                joint=joint_name,
                kp=(
                    motors_ctrl[joint_name].gains[0]*units.torques
                    if animat_options and motors_ctrl[joint_name].gains
                    else 0
                ),
                ctrllimited=act_pos_ctrllimited,
                ctrlrange=act_pos_ctrlrange,
                forcelimited=act_pos_forcelimited,
                forcerange=[val*units.torques for val in act_pos_forcerange],
            )
            name_vel = f'actuator_velocity_{joint_name}'
            mjcf_map['actuators'][name_vel] = mjcf_model.actuator.add(
                'velocity',
                name=name_vel,
                joint=joint_name,
                kv=(
                    motors_ctrl[joint_name].gains[1]*units.angular_damping
                    if animat_options and motors_ctrl[joint_name].gains
                    else 0
                ),
                ctrllimited=act_vel_ctrllimited,
                ctrlrange=[val*units.angular_velocity for val in act_vel_ctrlrange],
                forcelimited=act_vel_forcelimited,
                forcerange=[val*units.torques for val in act_vel_forcerange],
            )
            name_trq = f'actuator_torque_{joint_name}'
            mjcf_map['actuators'][name_trq] = mjcf_model.actuator.add(
                'motor',
                name=name_trq,
                joint=joint_name,
            )
            if (
                    animat_options is not None
                    and motors_ctrl[joint_name].limits_torque is not None
            ):
                torque_limits = [
                    trq*units.torques
                    for trq in motors_ctrl[joint_name].limits_torque
                ]
                for name in [name_pos, name_vel, name_trq]:
                    mjcf_map['actuators'][name].forcelimited = True
                    mjcf_map['actuators'][name].forcerange = torque_limits
        assert mjcf_map['actuators'], mjcf_map['actuators']

        # Muscles
        if use_muscles:
            # Add sites from muscle config file
            for muscle in animat_options.control.hill_muscles:
                # Add tendon
                tendon_name = f'{muscle.name.lower()}'
                mjcf_map['tendons'][tendon_name] = mjcf_model.tendon.add(
                    "spatial",
                    name=tendon_name,
                    group=4,
                    width=1e-3,
                    rgba=[1.0, 0.0, 0.0, 1],
                )
                # Add actuator
                muscle_name = f'{muscle.name.lower()}'
                prms = [
                    muscle['max_force']*units.newtons,
                    muscle['optimal_fiber']*units.meters,
                    muscle['tendon_slack']*units.meters,
                    muscle['max_velocity']*units.velocity, # vmax
                    np.deg2rad(muscle['pennation_angle']),
                ]
                mjcf_map['muscles'][muscle_name] = mjcf_model.actuator.add(
                    "general",
                    name=muscle_name,
                    tendon=tendon_name,
                    lengthrange=[
                        muscle['lmtu_min']*units.meters,
                        muscle['lmtu_max']*units.meters,
                    ],
                    forcelimited=True,
                    forcerange=[
                        -2*muscle['max_force']*units.newtons,
                        2*muscle['max_force']*units.newtons,
                    ],
                    dyntype='muscle',
                    gaintype='user',
                    biastype='user',
                    dynprm=[
                        0.01*units.seconds, 0.04*units.seconds
                    ], # act-deact time constants
                    gainprm=prms,
                    biasprm=prms,
                    user=[
                        # Type Ia
                        muscle['type_I_kv'],
                        muscle['type_I_pv'],
                        muscle['type_I_k_dI'],
                        muscle['type_I_k_nI'],
                        muscle['type_I_const_I'],
                        # Type II
                        muscle['type_II_k_dII'],
                        muscle['type_II_k_nII'],
                        muscle['type_II_const_II'],
                        # Type Ib
                        muscle['type_Ib_kF'],
                    ],
                )
                # Define waypoints
                for pindex, waypoint in enumerate(muscle['waypoints']):
                    body_name = waypoint[0]
                    position = [pos*units.meters for pos in waypoint[1]]
                    # Add sites
                    body = mjcf_model.worldbody.find('body', body_name)
                    site_name = f'{muscle_name}_P{pindex}'
                    body.add(
                        'site',
                        name=site_name,
                        pos=position,
                        group=3,
                        size=[1e-3*units.meters]*3,
                        rgba=[1.0, 0, 0, 1]
                    )
                    # Attach site to tendon
                    mjcf_map['tendons'][tendon_name].add(
                        'site', site=site_name
                    )

    # Sensors
    if use_sensors:

        for link_name in mjcf_map['links']:
            if use_link_sensors:
                for link_sensor in ('framepos', 'framequat'):
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
            if use_link_vel_sensors:
                for link_sensor in ('framelinvel', 'frameangvel'):
                    mjcf_model.sensor.add(
                        link_sensor,
                        name=f'{link_sensor}_{link_name}',
                        objname=link_name,
                        objtype='body',
                    )
        if use_joint_sensors:
            for joint_name in mjcf_map['joints']:
                for joint_sensor in ('jointpos', 'jointvel'):
                    mjcf_model.sensor.add(
                        joint_sensor,
                        name=f'{joint_sensor}_{joint_name}',
                        joint=joint_name,
                    )

        if use_actuator_sensors:
            for actuator_name, actuator in mjcf_map['actuators'].items():
                mjcf_model.sensor.add(
                    'actuatorfrc',
                    # Adapted for muscles
                    name=f'actuatorfrc_{actuator.tag}_{actuator.joint}',
                    actuator=actuator_name,
                )
        if use_muscle_sensors:
            for tendon_name, tendon in mjcf_map['tendons'].items():
                mjcf_model.sensor.add(
                    'tendonpos',
                    name=f'tendonpos_{tendon.name}',
                    tendon=tendon_name
                )
                mjcf_model.sensor.add(
                    'tendonvel',
                    name=f'tendonvel_{tendon.name}',
                    tendon=tendon_name
                )
            for muscle_name, muscle in mjcf_map['muscles'].items():
                mjcf_model.sensor.add(
                    'actuatorfrc',
                    name=f'musclefrc_{muscle_name}',
                    actuator=muscle_name,
                )

    # Contacts
    if animat_options is not None:
        collision_map = {
            link.name: [col.name for col in link.collisions]
            for link in sdf.links
        }
        pair_options = {}
        if solref is not None:
            pair_options['solref'] = solref
        for pair_i, (link1, link2) in enumerate(
                animat_options.morphology.self_collisions
        ):
            for col1_i, col1_name in enumerate(collision_map[link1]):
                for col2_i, col2_name in enumerate(collision_map[link2]):
                    mjcf_model.contact.add(
                        'pair',
                        name=f'contact_pair_{pair_i}_{col1_i}_{col2_i}',
                        geom1=col1_name,
                        geom2=col2_name,
                        condim=3,
                        friction=[0]*5,
                        **pair_options,
                    )

    return mjcf_model, mjcf_map


def mjcf2str(
        mjcf_model: mjcf.RootElement,
        remove_temp: bool = True,
) -> str:
    """Export to MJCF string"""
    # XML
    mjcf_xml = mjcf_model.to_xml()
    # Remove unique identifiers from mesh paths
    if remove_temp:
        for mesh in mjcf_xml.find('asset').findall('mesh'):
            mjcf_mesh = mjcf_model.find('mesh', mesh.attrib['name'])
            mesh.attrib['file'] = mjcf_mesh.file.prefix + mjcf_mesh.file.extension
    # Convert to string
    xml_str = ET.tostring(
        mjcf_xml,
        encoding='utf8',
        method='xml',
    ).decode('utf8')
    dom = xml.dom.minidom.parseString(xml_str)
    # for asset in mjcf_model.asset
    return dom.toprettyxml(indent=2*' ')


def night_sky(mjcf_model: mjcf.RootElement):
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


def add_plane(
        mjcf_model: mjcf.RootElement,
) -> (mjcf.RootElement, mjcf.RootElement, mjcf.RootElement):
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


def add_particles(mjcf_model: mjcf.RootElement):
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


def add_lights(link: mjcf.RootElement, rot: NDARRAY_3 = None):
    """Add lights"""
    if rot is None:
        rot = [0, 0, 0]
    rot_inv = Rotation.from_euler(angles=rot, seq='xyz').inv()
    link.add(
        'light',
        name='light_animat',
        mode='trackcom',  # 'targetbody'
        active=str(True).lower(),
        pos=rot_inv.as_matrix() @ [-1, 1, 3],
        dir=rot_inv.as_matrix() @ [1.0, -1.0, -3.0],
        castshadow=str(True).lower(),
        directional=str(False).lower(),
        attenuation=[1.0, 0.0, 0.0],
        cutoff=100,
        exponent=1.0,
        ambient=[0.0, 0.0, 0.0],
        diffuse=[0.7, 0.7, 0.7],
        specular=[0.3, 0.3, 0.3],
    )


def add_cameras(
        link: mjcf.RootElement,
        dist: float = 3,
        rot: NDARRAY_3 = None,
        simulation_options: SimulationOptions = None,
):
    """Add cameras"""
    if rot is None:
        rot = [0, 0, 0]
    rot_inv = Rotation.from_euler(angles=rot, seq='xyz').inv()
    if simulation_options is not None:
        dist = simulation_options.video_distance
        pitch = np.deg2rad(-90+simulation_options.video_pitch)
        yaw = np.deg2rad(-simulation_options.video_yaw)
        sim_options_camera = [['trackcom', [
            dist*np.sin(pitch)*np.sin(yaw) + simulation_options.video_offset[0],
            dist*np.sin(pitch)*np.cos(yaw) + simulation_options.video_offset[1],
            dist*np.cos(pitch) + simulation_options.video_offset[2],
            -pitch, 0, -yaw,
        ]]]
    else:
        sim_options_camera = []
    for i, (mode, pose) in enumerate(sim_options_camera + [
            ['trackcom', [0.0, 0.0, dist, 0.0, 0.0, 0.0]],
            ['trackcom', [0.0, -dist, 0.2*dist, 0.4*np.pi, 0.0, 0.0]],
            ['trackcom', [-dist, 0.0, 0.2*dist, 0.4*np.pi, 0, -0.5*np.pi]],
            # ['targetbodycom', [dist, 0.0, 0.2*dist, 0, 0, 0]],
    ]):
        link.add(
            'camera',
            name=f'camera_{link.name}_{i}',
            mode=mode,
            pos=rot_inv.as_matrix() @ pose[:3],
            quat=quat2mjcquat((
                rot_inv
                * Rotation.from_euler(angles=pose[3:], seq='xyz')
            ).as_quat()),
            # target=link.name,
            fovy=70,
            ipd=0.068,
        )


def setup_mjcf_xml(**kwargs) -> (mjcf.RootElement, mjcf.RootElement, Dict):
    """Setup MJCF XML"""

    hfield = None
    mjcf_model = None
    simulation_options = kwargs.pop('simulation_options', None)
    animat_options = kwargs.pop('animat_options', None)
    arena_options = kwargs.pop('arena_options', None)
    units = kwargs.pop('units', (
        simulation_options.units
        if simulation_options is not None
        else SimulationUnitScaling()
    ))
    timestep = kwargs.pop(
        'timestep',
        simulation_options.timestep/max(1, simulation_options.num_sub_steps)
        if simulation_options is not None
        else 1e-3,
    )

    # Arena
    mjcf_model, info = sdf2mjcf(
        sdf=ModelSDF.read(filename=os.path.expandvars(arena_options.sdf))[0],
        mjcf_model=mjcf_model,
        model_name='arena',
        fixed_base=True,
        concave=False,
        simulation_options=simulation_options,
        friction=[0, 0, 0],
        all_collisions=True,
    )
    if 'hfield' in info:
        hfield = info['hfield']
    arena_base_link = mjcf_model.worldbody.body[-1]
    arena_pose = arena_options.spawn.pose
    arena_base_link.pos = [pos*units.meters for pos in arena_pose[:3]]
    arena_base_link.quat = euler2mjcquat(euler=arena_pose[3:])
    if arena_options.ground_height is not None:
        arena_base_link.pos[2] += arena_options.ground_height*units.meters
    if arena_options.water.height is not None:
        mjcf_model, info = sdf2mjcf(
            sdf=ModelSDF.read(arena_options.water.sdf)[0],
            mjcf_model=mjcf_model,
            model_name='water',
            fixed_base=True,
            concave=False,
            simulation_options=simulation_options,
            friction=[0, 0, 0],
        )
        water = mjcf_model.worldbody.body[-1]
        water.pos = [0, 0, arena_options.water.height*units.meters]
        water.quat = [1, 0, 0, 0]
    # add_plane(mjcf_model)

    # Animat
    mujoco_kwargs = animat_options.mujoco if animat_options is not None else {}
    sdf_animat = ModelSDF.read(os.path.expandvars(animat_options.sdf))[0]
    mjcf_model, _ = sdf2mjcf(
        sdf=sdf_animat,
        mjcf_model=mjcf_model,
        model_name=animat_options.name,
        use_sensors=True,
        use_link_sensors=False,
        use_link_vel_sensors=True,
        use_joint_sensors=False,
        use_actuators=True,
        animat_options=animat_options,
        simulation_options=simulation_options,
        **mujoco_kwargs,
    )
    base_link = mjcf_model.worldbody.body[-1]
    base_link.pos = animat_options.spawn.pose[:3]
    base_link.quat = euler2mjcquat(animat_options.spawn.pose[3:])

    # Compiler
    mjcf_model.compiler.angle = 'radian'
    mjcf_model.compiler.eulerseq = 'xyz'
    mjcf_model.compiler.boundmass = MIN_MASS*units.kilograms
    mjcf_model.compiler.boundinertia = MIN_INERTIA*units.inertia
    mjcf_model.compiler.balanceinertia = False
    mjcf_model.compiler.inertiafromgeom = False
    mjcf_model.compiler.convexhull = True
    mjcf_model.compiler.fusestatic = True
    mjcf_model.compiler.discardvisual = kwargs.pop(
        'discardvisual',
        simulation_options.headless and not simulation_options.video
        if simulation_options is not None
        else False
    )
    # Disable lengthrange computation for muscles
    mjcf_model.compiler.lengthrange.mode = "none"
    mjcf_model.compiler.lengthrange.useexisting = True

    # Statistic
    scale = (
        1.0
        if not simulation_options
        else simulation_options.visual_scale
    )
    mjcf_model.statistic.meansize = 1
    mjcf_model.statistic.meanmass = 1
    mjcf_model.statistic.meaninertia = 1
    # mjcf_model.statistic.extent = 100*units.meters

    # Visual
    mjcf_model.visual.map.stiffness = 100
    mjcf_model.visual.map.stiffnessrot = 500
    mjcf_model.visual.map.force = 1*units.meters/units.newtons*scale
    mjcf_model.visual.map.torque = 1*units.meters/units.torques*scale
    mjcf_model.visual.map.znear = 1e-4*units.meters
    mjcf_model.visual.map.zfar = 3e0*units.meters
    mjcf_model.visual.map.alpha = 0.3
    mjcf_model.visual.map.fogstart = 3
    mjcf_model.visual.map.fogend = 10
    mjcf_model.visual.map.haze = 0.3
    mjcf_model.visual.map.shadowclip = 1
    mjcf_model.visual.map.shadowscale = 3e-1
    mjcf_model.visual.map.actuatortendon = 2*scale
    mjcf_model.visual.scale.forcewidth = 0.01*scale
    mjcf_model.visual.scale.contactwidth = 0.03*scale
    mjcf_model.visual.scale.contactheight = 0.01*scale
    mjcf_model.visual.scale.connect = 0.01*scale
    mjcf_model.visual.scale.com = 0.01*scale
    mjcf_model.visual.scale.camera = 0.1*scale
    mjcf_model.visual.scale.light = 0.1*scale
    mjcf_model.visual.scale.selectpoint = 0.01*scale
    mjcf_model.visual.scale.jointlength = 0.01*scale
    mjcf_model.visual.scale.jointwidth = 0.01*scale
    mjcf_model.visual.scale.actuatorlength = 0.01*scale
    mjcf_model.visual.scale.actuatorwidth = 0.01*scale
    mjcf_model.visual.scale.framelength = 0.01*scale
    mjcf_model.visual.scale.framewidth = 0.01*scale
    mjcf_model.visual.scale.constraint = 0.01*scale
    mjcf_model.visual.scale.slidercrank = 0.01*scale
    mjcf_model.visual.quality.shadowsize = 1024  # 32*1024
    mjcf_model.visual.quality.offsamples = 4
    mjcf_model.visual.quality.numslices = 28
    mjcf_model.visual.quality.numstacks = 16
    mjcf_model.visual.quality.numquads = 4
    glob = mjcf_model.visual.get_children('global')  # Global reserved in Python
    glob.offwidth = (
        simulation_options.video_resolution[0]
        if simulation_options is not None
        else 1280
    )
    glob.offheight = (
        simulation_options.video_resolution[1]
        if simulation_options is not None
        else 720
    )

    # Simulation options
    mjcf_model.size.njmax = 2**12  # 4096
    mjcf_model.size.nconmax = 2**12  # 4096
    mjcf_model.option.timestep = timestep
    mjcf_model.option.gravity = kwargs.pop(
        'gravity',
        [gravity*units.acceleration for gravity in simulation_options.gravity]
        if simulation_options is not None
        else [0, 0, -9.81]
    )
    mjcf_model.option.iterations = kwargs.pop(
        'solver_iterations',
        simulation_options.n_solver_iters
        if simulation_options is not None
        else 1000,
    )
    mjcf_model.option.solver = kwargs.pop('solver', 'Newton')  # PGS, CG
    mjcf_model.option.integrator = kwargs.pop('integrator', 'Euler')  # RK4
    mjcf_model.option.mpr_iterations = kwargs.pop('mpr_iterations', 100)  # 50
    mjcf_model.option.noslip_iterations = kwargs.pop('npslip_iterations', 100)
    mjcf_model.option.noslip_tolerance = kwargs.pop('npslip_tolerance', 1e-8)
    mjcf_model.option.tolerance = kwargs.pop('tolerance', 1e-12)
    mjcf_model.option.cone = (
        simulation_options.pop('cone')
        if simulation_options is not None
        else 'pyramidal'
    )

    # Animat options
    if animat_options is not None:

        # Spawn
        animat_spawn = animat_options.spawn
        base_link.pos = [pos*units.meters for pos in animat_spawn.pose[:3]]
        base_link.quat = euler2mjcquat(animat_spawn.pose[3:])

        # Links
        for link in animat_options.morphology.links:
            mjcf_link = mjcf_model.find(namespace='body', identifier=link.name)
            assert mjcf_link, f'Link {link.name} not found'
            for geom in mjcf_link.geom:
                if geom.contype:
                    assert len(link.friction) == 3, len(geom.friction)
                    assert len(geom.friction) == 3, len(geom.friction)
                    geom.friction = link.friction
                    geom.fluidshape = None
                    geom.fluidcoef = [0, 0, 0, 0, 0]

        # Joints
        for joint_options in animat_options.morphology.joints:
            joint = mjcf_model.find(
                namespace='joint',
                identifier=joint_options.name,
            )
            joint.stiffness += joint_options.stiffness*units.angular_stiffness
            joint.damping += joint_options.damping*units.angular_damping
            if _solreflimit := joint_options.extras.get('solreflimit'):
                if all(sol < 0 for sol in _solreflimit):
                    _solreflimit[0] *= units.newtons/units.meters
                    _solreflimit[1] *= units.newtons/units.velocity
                else:
                    _solreflimit[0] *= units.seconds
                joint.solreflimit = joint_options.extras['solreflimit']
            if _solimplimit := joint_options.extras.get('solimplimit'):
                joint.solimplimit = _solimplimit
            if _margin := joint_options.extras.get('margin'):
                joint.margin = _margin # radians

        # Joints control
        joints_equations = {}
        for motor_options in animat_options.control.motors:
            if not hasattr(motor_options, 'equation'):
                continue
            joint = mjcf_model.find(
                namespace='joint',
                identifier=motor_options.joint_name,
            )
            joints_equations[motor_options.joint_name] = motor_options.equation
            if motor_options.passive.is_passive:
                joint.stiffness += (
                    motor_options.passive.stiffness_coefficient
                )*units.angular_stiffness
                joint.damping += (
                    motor_options.passive.damping_coefficient
                )*units.angular_damping

        # Muscles
        if (
                hasattr(animat_options.control, 'muscles')
                and animat_options.control.muscles is not None
        ):
            for muscle_options in animat_options.control.muscles:
                joint = mjcf_model.find(
                    namespace='joint',
                    identifier=muscle_options.joint_name,
                )
                assert joint, f'Joint {muscle_options.joint_name} not found'
                if 'ekeberg' in joints_equations[muscle_options.joint_name]:
                    joint.stiffness += (
                        muscle_options.beta*muscle_options.gamma
                    )*units.angular_stiffness
                    joint.damping += (
                        muscle_options.delta
                    )*units.angular_damping


    # Add particles
    if kwargs.pop('use_particles', False):
        add_particles(mjcf_model)

    # Light and shadows
    add_lights(link=base_link, rot=animat_options.spawn.pose[3:])

    # Add cameras
    add_cameras(
        link=base_link,
        rot=animat_options.spawn.pose[3:],
        simulation_options=simulation_options,
    )

    # Night sky
    night_sky(mjcf_model)

    # XML string
    mjcf_xml_str = mjcf2str(mjcf_model=mjcf_model)
    if kwargs.pop('show_mjcf', False):
        pylog.info(mjcf_xml_str)
    if kwargs.pop('save_mjcf', False):
        with open('simulation_mjcf.xml', 'w+', encoding='utf-8') as xml_file:
            xml_file.write(mjcf_xml_str)

    assert not kwargs, kwargs
    return mjcf_model, base_link, hfield
