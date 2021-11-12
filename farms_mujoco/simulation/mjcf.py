"""MJCF handling"""

import os
import xml.dom.minidom
import xml.etree.ElementTree as ET

import numpy as np
import trimesh as tri
from scipy.spatial.transform import Rotation

from dm_control import mjcf

import farms_pylog as pylog
from farms_sdf.sdf import (
    ModelSDF, Mesh, Visual, Collision,
    Box, Cylinder, Capsule, Sphere,
)

MIN_MASS = 1e-12
MIN_INERTIA = 1e-12


def poseul2mat4d(position, euler):
    """4D transform"""
    transform = np.eye(4)
    transform[:3, -1] = position
    transform[:3, :3] = Rotation.from_euler(
        angles=euler,
        seq='xyz',
    ).as_matrix()
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


def euler2mcjquat(euler):
    """Euler to Mujoco quaternion"""
    return Rotation.from_euler(
        angles=euler,
        seq='xyz',
    ).as_quat()[[3, 0, 1, 2]]


def mjc_add_link(mjcf_model, mjcf_items, sdf_link, **kwargs):
    """Add link to world"""

    sdf_parent = kwargs.pop('sdf_parent', None)
    mjc_parent = kwargs.pop('mjc_parent', None)
    sdf_joint = kwargs.pop('sdf_joint', None)
    directory = kwargs.pop('directory', '')
    free = kwargs.pop('free', False)

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
        pos=link_local_pos,
        quat=euler2mcjquat(link_local_euler),
    )
    mjcf_items['links'][sdf_link.name] = body

    # Site
    if kwargs.pop('use_site', False):
        site = body.add(
            'site',
            type='box',
            name=f'site_{sdf_link.name}',
            group=1,
            pos=[0, 0, 0],
            quat=[1, 0, 0, 0],
            size=[1e-2]*3,
        )
        mjcf_items['sites'][site.name] = site

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
                pos=sdf_joint.pose[:3],
                # euler=sdf_joint.pose[3:],  # Euler not supported in joint
                damping=1e-3,
                frictionloss=1e-3,
            )
            mjcf_items['joints'][sdf_joint.name] = joint

    # Visual and collisions (geoms)
    for element in sdf_link.collisions + sdf_link.visuals:

        geom = None

        # Properties
        self_collisions = 0
        friction = [0.5, 0, 0]

        # Include in mjcf
        visual_kwargs = {}
        collision_kwargs = {}
        geom_kwargs = {
            'name': element.name,
            'pos': element.pose[:3],
            'quat': euler2mcjquat(element.pose[3:]),
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
            collision_kwargs['contype'] = 1  # World collisions
            collision_kwargs['conaffinity'] = self_collisions
            collision_kwargs['group'] = 2

        # Mesh
        if isinstance(element.geometry, Mesh):

            # Mesh path
            mesh_path = os.path.join(directory, element.geometry.uri)
            assert os.path.isfile(mesh_path)

            # Convert to STL if mesh in other format
            path, extension = os.path.splitext(mesh_path)
            if extension != '.stl':
                stl_path = f'{path}.stl'
                if not os.path.isfile(stl_path):
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

            # Add mesh asset
            mjcf_model.asset.add(  # Add mesh to assets
                'mesh',
                name=f'mesh_{element.name}',
                file=mesh_path,
                scale=element.geometry.scale,
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
                size=[0.5*s for s in element.geometry.size],
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
                    element.geometry.radius,
                    0.5*element.geometry.length,
                    element.geometry.radius,
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
                    element.geometry.radius,
                    max(
                        element.geometry.length-2*element.geometry.radius,
                        element.geometry.radius,
                    ),
                    element.geometry.radius,
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
                size=[element.geometry.radius]*3,
                **geom_kwargs,
                **visual_kwargs,
                **collision_kwargs,
            )

        else:
            raise NotImplementedError(f'{type(element.geometry)} not supported')

        if geom is not None:
            if isinstance(element, Visual):
                mjcf_items['visuals'][element.name] = geom
            elif isinstance(element, Collision):
                mjcf_items['collisions'][element.name] = geom

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
        eigvals = np.linalg.eigvals(inertia_mat)
        assert (eigvals > 0).all(), f'Eigen values <= 0: {eigvals}\n{inertia_mat}'
        body.add(
            'inertial',
            pos=inertial.pose[:3],
            quat=euler2mcjquat(inertial.pose[3:]),
            mass=inertial.mass,
            # diaginertia=np.clip([
            #     inertial.inertias[0],
            #     inertial.inertias[3],
            #     inertial.inertias[5]
            # ], 0, np.inf).tolist(),
            fullinertia=[
                max(MIN_INERTIA, inertial.inertias[0]),
                max(MIN_INERTIA, inertial.inertias[3]),
                max(MIN_INERTIA, inertial.inertias[5]),
                inertial.inertias[1],
                inertial.inertias[2],
                inertial.inertias[4],
            ],
        )

    return body, joint


def add_link_recursive(mjcf_model, mjcf_items, sdf, **kwargs):
    """Add link recursive"""

    # Handle kwargs
    sdf_link = kwargs.pop('sdf_link', None)
    sdf_parent = kwargs.pop('sdf_parent', None)
    sdf_joint = kwargs.pop('sdf_joint', None)
    use_site = kwargs.pop('use_site', False)

    # Add link
    mjc_add_link(
        mjcf_model=mjcf_model,
        mjcf_items=mjcf_items,
        sdf_link=sdf_link,
        sdf_parent=sdf_parent,
        sdf_joint=sdf_joint,
        directory=sdf.directory,
        use_site=use_site,
        mjc_parent=(
            mjcf_items['links'].get(sdf_parent.name)
            if sdf_parent is not None
            else None
        ),
        **kwargs,
    )

    # Add children
    for child in sdf.get_children(link=sdf_link):
        add_link_recursive(
            mjcf_model=mjcf_model,
            mjcf_items=mjcf_items,
            sdf=sdf,
            sdf_link=child,
            sdf_parent=sdf_link,
            sdf_joint=sdf.get_parent_joint(link=child),
            use_site=use_site,
        )


def sdf2mjcf(sdf, mjcf_model=None, use_site=False):
    """Export to MJCF string"""

    if mjcf_model is None:
        mjcf_model = mjcf.RootElement()

    # Compiler
    mjcf_model.compiler.angle = 'radian'
    mjcf_model.compiler.eulerseq = 'xyz'
    mjcf_model.compiler.boundmass = MIN_MASS
    mjcf_model.compiler.boundinertia = MIN_INERTIA
    mjcf_model.compiler.balanceinertia = True
    mjcf_model.compiler.inertiafromgeom = False

    # Base link
    base_link = sdf.get_base_link()

    # Links and joints
    mjcf_items = {
        element: {}
        for element in ['links', 'joints', 'sites', 'visuals', 'collisions']
    }

    add_link_recursive(
        mjcf_model=mjcf_model,
        mjcf_items=mjcf_items,
        sdf=sdf,
        sdf_link=base_link,
        free=True,
        use_site=use_site,
    )

    # Actuators
    for joint_name in mjcf_items['joints']:
        mjcf_model.actuator.add(
            'position',
            name=f'actuator_position_{joint_name}',
            joint=joint_name,
            kp=1e-2,
        )
        mjcf_model.actuator.add(
            'velocity',
            name=f'actuator_velocity_{joint_name}',
            joint=joint_name,
            kv=1e-3,
        )

    # Sensors
    for link_name in mjcf_items['links']:
        for link_sensor in ('framepos', 'framequat', 'framelinvel', 'frameangvel'):
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
    for joint_name in mjcf_items['joints']:
        for joint_sensor in ('jointpos', 'jointvel'):
            mjcf_model.sensor.add(
                joint_sensor,
                name=f'{joint_sensor}_{joint_name}',
                joint=joint_name,
            )
        for joint_sensor in ('actuatorfrc',):
            for actuator_type in ('position', 'velocity'):
                mjcf_model.sensor.add(
                    joint_sensor,
                    name=f'{joint_sensor}_{actuator_type}_{joint_name}',
                    actuator=f'actuator_{actuator_type}_{joint_name}',
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
    material = mjcf_model.asset.add(  # Add grid material to assets
        'material',
        name='material_grid',
        texture='texture_grid',
        texrepeat=[1, 1],
        texuniform=True,
        reflectance=0.2,
    )
    geometry = mjcf_model.worldbody.add(  # Add floor
        'geom',
        type='plane',
        name='floor',
        pos=[0, 0, 0],
        size=[10, 10, 0.1],
        material='material_grid',
        friction=[1, 0, 0],
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


def add_lights(mjcf_model, link):
    """Add lights"""
    mjcf_model.visual.quality.shadowsize = 32*1024
    for i, pos in enumerate([[-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]]):
        link.add(
            'light',
            name=f'light_{link.name}_{i}',
            mode='trackcom',
            # target='targetbody',
            active=str(True).lower(),
            pos=pos,
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


def add_cameras(mjcf_model, link, dist=0.3):
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
            pos=pose[:3],
            quat=euler2mcjquat(pose[3:]),
            # target='',
            fovy=45,
            ipd=0.068,
        )


def setup_mjcf_xml(sdf_path, timestep, **kwargs):
    """Setup MJCF XML"""

    # Animat
    sdf = ModelSDF.read(filename=os.path.expandvars(sdf_path))[0]
    mjcf_model = sdf2mjcf(sdf)
    mjcf_model.size.njmax = 2**12
    mjcf_model.size.nconmax = 2**12
    mjcf_model.option.timestep = timestep
    mjcf_model.option.iterations = kwargs.pop('solver_iterations', 100)
    mjcf_model.option.solver = 'Newton'  # PGS, CG, Newton
    mjcf_model.option.integrator = 'Euler'  # Euler, RK4
    mjcf_model.option.mpr_iterations = kwargs.pop('mpr_iterations', 50)

    # Spawn options
    base_link = mjcf_model.worldbody.body[0]
    base_link.pos = kwargs.pop('spawn_position', [0, 0, 0])
    base_link.euler = kwargs.pop('spawn_rotation', [0, 0, 0])

    # Add plane
    add_plane(mjcf_model)

    # Add particles
    if kwargs.pop('use_particles', False):
        add_particles(mjcf_model)

    # Light and shadows
    add_lights(mjcf_model=mjcf_model, link=base_link)

    # Add cameras
    add_cameras(mjcf_model=mjcf_model, link=base_link)

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
