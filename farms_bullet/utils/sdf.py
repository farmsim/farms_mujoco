"""SDF"""

import os
import numpy as np
from scipy.spatial.transform import Rotation
import pybullet
import farms_pylog as pylog
from farms_sdf.sdf import (
    ModelSDF,
    Plane,
    Box,
    Sphere,
    Cylinder,
    Capsule,
    Mesh,
    Heightmap,
    Collision,
)
from ..utils.output import redirect_output


def rot_quat(rot):
    """Quaternion from Euler"""
    return pybullet.getQuaternionFromEuler(rot)


def rot_matrix(rot):
    """Matrix from Quaternion"""
    return np.array(pybullet.getMatrixFromQuaternion(rot)).reshape([3, 3])


def rot_invert(rot):
    """Invert rot"""
    return pybullet.invertTransform([0, 0, 0], rot)[1]


def rot_mult(rot0, rot1):
    """Rotation Multiplication"""
    return pybullet.multiplyTransforms([0, 0, 0], rot0, [0, 0, 0], rot1)[1]


def rot_diff(rot0, rot1):
    """Rotation difference"""
    return pybullet.getDifferenceQuaternion(rot0, rot1)


def pybullet_options_from_shape(shape, path='', force_concave=False, meters=1):
    """Pybullet shape"""
    options = {}
    collision = isinstance(shape, Collision)
    if collision:
        options['collisionFramePosition'] = np.array(shape.pose[:3])*meters
        options['collisionFrameOrientation'] = rot_quat(shape.pose[3:])
    else:
        options['visualFramePosition'] = np.array(shape.pose[:3])*meters
        options['visualFrameOrientation'] = rot_quat(shape.pose[3:])
        options['rgbaColor'] = shape.diffuse
        options['specularColor'] = shape.specular
    if isinstance(shape.geometry, Plane):
        options['shapeType'] = pybullet.GEOM_PLANE
        options['planeNormal'] = shape.geometry.normal
    elif isinstance(shape.geometry, Box):
        options['shapeType'] = pybullet.GEOM_BOX
        options['halfExtents'] = 0.5*np.array(shape.geometry.size)*meters
    elif isinstance(shape.geometry, Sphere):
        options['shapeType'] = pybullet.GEOM_SPHERE
        options['radius'] = shape.geometry.radius*meters
    elif isinstance(shape.geometry, Cylinder):
        options['shapeType'] = pybullet.GEOM_CYLINDER
        options['radius'] = shape.geometry.radius*meters
        options['height' if collision else 'length'] = (
            shape.geometry.length*meters
        )
    elif isinstance(shape.geometry, Capsule):
        options['shapeType'] = pybullet.GEOM_CAPSULE
        options['radius'] = shape.geometry.radius*meters
        options['height' if collision else 'length'] = (
            shape.geometry.length*meters
        )
    elif isinstance(shape.geometry, Mesh):
        options['shapeType'] = pybullet.GEOM_MESH
        options['fileName'] = os.path.join(path, shape.geometry.uri)
        options['meshScale'] = np.array(shape.geometry.scale)*meters
        if force_concave:
            options['flags'] = pybullet.GEOM_FORCE_CONCAVE_TRIMESH
    elif isinstance(shape.geometry, Heightmap):
        options['shapeType'] = pybullet.GEOM_HEIGHTFIELD
    else:
        raise Exception('Unknown type {}'.format(type(shape.geometry)))
    return options


def find_joint(sdf_model, link):
    """Find joint"""
    for joint in sdf_model.joints:
        if joint.child == link.name:
            return joint
    return None


def joint_pybullet_type(joint):
    """Find joint"""
    if joint.type in ('revolute', 'continuous'):
        return pybullet.JOINT_REVOLUTE
    return pybullet.JOINT_FIXED


def reset_controllers(identity):
    """Reset controllers"""
    n_joints = pybullet.getNumJoints(identity)
    joints = np.arange(n_joints)
    zeros = np.zeros_like(joints)
    pybullet.setJointMotorControlArray(
        identity,
        joints,
        pybullet.POSITION_CONTROL,
        targetPositions=zeros,
        targetVelocities=zeros,
        forces=zeros
    )
    pybullet.setJointMotorControlArray(
        identity,
        joints,
        pybullet.VELOCITY_CONTROL,
        targetVelocities=zeros,
        forces=zeros,
    )
    pybullet.setJointMotorControlArray(
        identity,
        joints,
        pybullet.TORQUE_CONTROL,
        forces=zeros
    )


def rearange_base_link_list(table, base_link_index):
    """Rarange base link to beginning of table"""
    value = table[base_link_index]
    del table[base_link_index]
    table.insert(0, value)
    return table


def rearange_base_link_dict(dictionary, base_link_index):
    """Rarange base link to beginning of table"""
    new_dict = {}
    for key, value in dictionary.items():
        new_dict[key] = (
            0 if value == base_link_index
            else value + 1 if value < base_link_index
            else value
        )
    return new_dict


def load_sdf(
        sdf_path,
        force_concave=False,
        reset_control=True,
        links_options=None,
        use_self_collision=False,
        **kwargs,
):
    """Load SDF"""
    units = kwargs.pop('units')
    verbose = kwargs.pop('verbose', False)
    assert not kwargs, kwargs
    sdf = ModelSDF.read(sdf_path)[0]
    folder = os.path.dirname(sdf_path)
    links_names = []
    visuals = []
    collisions = []
    joint_types = []
    joints_names = []
    link_index = {}
    joints_axis = []
    link_pos = []
    link_ori = []
    link_masses = []
    link_com = []
    link_inertias = []
    link_inertiaori = []
    link_i = 0
    base_link_index = None
    parenting = {joint.child: joint.parent for joint in sdf.joints}
    for link in sdf.links:

        # Number of visuals/collisions in link
        n_links = max(1, len(link.visuals), len(link.collisions))

        # Joint information
        joint = find_joint(sdf, link)

        # Visual and collisions
        for i in range(n_links):
            # Visuals
            visuals.append(
                pybullet.createVisualShape(
                    **pybullet_options_from_shape(
                        link.visuals[i],
                        path=folder,
                        force_concave=force_concave,
                        meters=units.meters,
                    )
                ) if i < len(link.visuals) else -1
            )
            # Collisions
            collisions.append(
                pybullet.createCollisionShape(
                    **pybullet_options_from_shape(
                        link.collisions[i],
                        path=folder,
                        force_concave=force_concave,
                        meters=units.meters,
                    )
                ) if i < len(link.collisions) else -1
            )
            if i > 0:
                # Dummy links to support multiple visuals and collisions
                link_name = '{}_dummy_{}'.format(link.name, i-1)
                link_index[link_name] = link_i
                links_names.append(link_name)
                assert link.name in link_index, (
                    'Link {} is not in link_index {}'.format(
                        link.name,
                        link_index,
                    )
                )
                parenting[link_name] = link.name
                link_pos.append(link.pose[:3])
                link_ori.append(link.pose[3:])
                link_masses.append(0)
                link_com.append([0, 0, 0])
                link_inertias.append([0, 0, 0])
                link_inertiaori.append([0, 0, 0, 1])
                joint_types.append(pybullet.JOINT_FIXED)
                joints_names.append('joint_dummy_{}_{}'.format(link.name, i-1))
                joints_axis.append([0.0, 0.0, 1.0])
            else:
                # Link information
                link_index[link.name] = link_i
                links_names.append(link.name)
                link_pos.append(link.pose[:3])
                link_ori.append(link.pose[3:])
                if link.inertial is None:
                    link_masses.append(0)
                    link_com.append([0, 0, 0])
                    link_inertias.append([0, 0, 0])
                    link_inertiaori.append([0, 0, 0, 1])
                else:
                    link_masses.append(link.inertial.mass)
                    link_com.append(link.inertial.pose[:3])
                    inertia_vec = link.inertial.inertias
                    inertia_tensor = np.array([
                        [inertia_vec[0], inertia_vec[1], inertia_vec[2]],
                        [inertia_vec[1], inertia_vec[3], inertia_vec[4]],
                        [inertia_vec[2], inertia_vec[4], inertia_vec[5]],
                    ])
                    inertias, inertia_vectors = np.linalg.eig(inertia_tensor)
                    link_inertias.append(inertias)
                    link_inertiaori.append(
                        Rotation.from_matrix(inertia_vectors).as_quat()
                    )
                # Joint information
                if joint is None:
                    # Base link
                    assert base_link_index is None, 'Found two base links'
                    base_link_index = link_i
                else:
                    joint_types.append(joint_pybullet_type(joint))
                    joints_names.append(joint.name)
                    joints_axis.append(
                        joint.axis.xyz
                        if joint.axis is not None
                        else [0.0, 0.0, 1.0]
                    )
            link_i += 1
    n_links = link_i

    # Rearange base link at beginning
    links_names = rearange_base_link_list(links_names, base_link_index)
    link_pos = rearange_base_link_list(link_pos, base_link_index)
    link_ori = rearange_base_link_list(link_ori, base_link_index)
    link_com = rearange_base_link_list(link_com, base_link_index)
    link_masses = rearange_base_link_list(link_masses, base_link_index)
    link_inertias = rearange_base_link_list(link_inertias, base_link_index)
    link_inertiaori = rearange_base_link_list(link_inertiaori, base_link_index)
    visuals = rearange_base_link_list(visuals, base_link_index)
    collisions = rearange_base_link_list(collisions, base_link_index)
    link_index = rearange_base_link_dict(link_index, base_link_index)

    for name in links_names[1:]:
        assert name in parenting, (
            'Link \'{}\' not in {}'.format(
                name,
                parenting,
            )
        )
        assert parenting[name] in link_index, (
            'Link \'{}\' (Parent of \'{}\') not in {}\n\nParenting:\n{}'.format(
                parenting[name],
                name,
                link_index,
                parenting,
            )
        )
    link_parent_indices = [
        link_index[parenting[name]]
        for name in links_names[1:]
    ]

    if links_options:
        # Modify masses
        mass_multiplier_map = {
            link.name: link.mass_multiplier
            for link in links_options
        }
        link_masses = [
            mass_multiplier_map[link_name]*link_mass
            if link_name in mass_multiplier_map
            else link_mass
            for link_name, link_mass in zip(links_names, link_masses)
        ]
        link_inertias = [
            mass_multiplier_map[link_name]*link_inertia
            if link_name in mass_multiplier_map
            else link_inertia
            for link_name, link_inertia in zip(links_names, link_inertias)
        ]

    # Local information
    link_local_positions = []
    link_local_orientations = []
    for pos, name, ori in zip(link_pos, links_names, link_ori):
        if name in parenting:
            link_local_positions.append(
                pybullet.multiplyTransforms(
                    [0, 0, 0],
                    rot_invert(rot_quat(link_ori[link_index[parenting[name]]])),
                    (
                        np.array(pos)
                        - np.array(link_pos[link_index[parenting[name]]])
                    ),
                    rot_quat([0, 0, 0]),
                )[0]
            )
            link_local_orientations.append(
                rot_mult(
                    rot_invert(rot_quat(link_ori[link_index[parenting[name]]])),
                    rot_quat(ori),
                )
            )

    # Model information
    if verbose:
        pylog.debug('\n'.join(
            [
                (
                    '0 (Base link): {}'
                    ' - index: {}'
                    ' - mass: {:.4f} [g]'
                    ' - inertias: [{:.3e}, {:.3e}, {:.3e}]'
                ).format(
                    name,
                    link_i,
                    1e3*link_masses[link_i],
                    link_inertias[link_i][0],
                    link_inertias[link_i][1],
                    link_inertias[link_i][2],
                )
                if link_i == 0
                else (
                    '{: >3} {: <20}'
                    ' - parent: {: <20} ({: >2})'
                    ' - mass: {:.4f} [g]'
                    ' - inertias: [{:.3e} {:.3e} {:.3e}]'
                    ' - joint: {: <20} - axis: {}'
                ).format(
                    '{}:'.format(link_i),
                    name,
                    parenting[name],
                    link_index[parenting[name]],
                    1e3*link_masses[link_i],
                    link_inertias[link_i][0],
                    link_inertias[link_i][1],
                    link_inertias[link_i][2],
                    joints_names[link_i-1],
                    joints_axis[link_i-1],
                )
                for link_i, name in enumerate(links_names)
            ] + ['\nTotal mass: {:.4f} [g]'.format(1e3*sum(link_masses))]
        ))
        pylog.debug('Spawning model')

    # Spawn model
    identity = pybullet.createMultiBody(
        baseMass=link_masses[0],
        basePosition=np.array(link_pos[0])*units.meters,
        baseOrientation=rot_quat(link_ori[0]),
        baseVisualShapeIndex=visuals[0],
        baseCollisionShapeIndex=collisions[0],
        baseInertialFramePosition=np.array(link_com[0])*units.meters,
        baseInertialFrameOrientation=link_inertiaori[0],
        linkMasses=link_masses[1:],
        linkPositions=np.array(link_local_positions)*units.meters,
        linkOrientations=link_local_orientations,
        linkInertialFramePositions=np.array(link_com[1:])*units.meters,
        linkInertialFrameOrientations=link_inertiaori[1:],
        linkVisualShapeIndices=visuals[1:],
        linkCollisionShapeIndices=collisions[1:],
        linkParentIndices=link_parent_indices,
        linkJointTypes=joint_types,
        linkJointAxis=joints_axis,
        flags=(
            pybullet.URDF_USE_SELF_COLLISION
            | pybullet.URDF_MERGE_FIXED_LINKS
        ) if use_self_collision else (
            pybullet.URDF_MERGE_FIXED_LINKS
            # | pybullet.URDF_USE_SELF_COLLISION
            # | pybullet.URDF_MAINTAIN_LINK_ORDER  # Removes certain links?
            # | pybullet.URDF_ENABLE_SLEEPING
            # | pybullet.URDF_USE_INERTIA_FROM_FILE
            # | pybullet.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
            # | pybullet.URDF_PRINT_URDF_INFO
            # | pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL
        ),
    )

    # Reset control
    if reset_control:
        reset_controllers(identity)

    # Get links and joints maps
    links, joints = {}, {}
    links[links_names[0]] = -1
    for joint_i in range(pybullet.getNumJoints(identity)):
        joint_info = pybullet.getJointInfo(identity, joint_i)
        # Link
        link_name = joint_info[12].decode('UTF-8')
        link_number = int(link_name.replace('link', ''))
        links[links_names[link_number]] = joint_i
        # Joint
        joint_name = joint_info[1].decode('UTF-8')
        joint_number = int(joint_name.replace('joint', ''))-1
        joints[joints_names[joint_number]] = joint_i
        # Checks
        assert joint_types[joint_number] == joint_info[2]

    # Set inertias
    for link_name, inertia in zip(links_names, link_inertias):
        pybullet.changeDynamics(
            bodyUniqueId=identity,
            linkIndex=links[link_name],
            localInertiaDiagonal=inertia,
        )

    # Units scaling
    inertia_unit = units.kilograms*units.meters**2
    for link_name, link in links.items():
        mass, inertias = np.array(
            pybullet.getDynamicsInfo(identity, link),
            dtype=object,
        )[[0, 2]]
        assert np.isclose(mass, link_masses[links_names.index(link_name)])
        if joint.axis is not None and joint.axis.limits is not None:
            assert joint.axis.limits[0] < joint.axis.limits[1]
            pybullet.changeDynamics(
                bodyUniqueId=identity,
                linkIndex=link,
                jointLowerLimit=joint.axis.limits[0],
                jointUpperLimit=joint.axis.limits[1],
            )
        pybullet.changeDynamics(
            bodyUniqueId=identity,
            linkIndex=link,
            mass=mass*units.kilograms,
        )
        pybullet.changeDynamics(
            bodyUniqueId=identity,
            linkIndex=link,
            localInertiaDiagonal=np.array(inertias)*inertia_unit,
        )

    # Set joints properties
    for joint in sdf.joints:
        if joint.axis is not None and joint.axis.limits is not None:
            pybullet.changeDynamics(
                bodyUniqueId=identity,
                linkIndex=joints[joint.name],
                jointLowerLimit=joint.axis.limits[0],
                jointUpperLimit=joint.axis.limits[1],
            )
            for key, value in [
                    ['jointDamping', 0*units.torques*units.seconds],
                    ['jointLimitForce', joint.axis.limits[2]*units.newtons],
                    ['maxJointVelocity', joint.axis.limits[3]/units.seconds],
            ]:
                pybullet.changeDynamics(
                    bodyUniqueId=identity,
                    linkIndex=joints[joint.name],
                    **{key: value}
                )

    return identity, links, joints


def load_sdf_pybullet(sdf_path, index=0, morphology_links=None, **kwargs):
    """Original way of loading SDF - Deprecated"""
    units = kwargs.pop('units')
    assert not kwargs, kwargs
    links, joints = {}, {}
    with redirect_output(pylog.warning):
        identity = pybullet.loadSDF(
            sdf_path,
            useMaximalCoordinates=0,
            globalScaling=units.meters,
        )[index]
    inertia_unit = units.kilograms*units.meters**2
    for joint_i in range(pybullet.getNumJoints(identity)):
        joint_info = pybullet.getJointInfo(identity, joint_i)
        link_name = joint_info[12].decode('UTF-8')
        links[link_name] = joint_i
        joints[joint_info[1].decode('UTF-8')] = joint_i
        mass, inertias = np.array(
            pybullet.getDynamicsInfo(identity, links[link_name]),
            dtype=object,
        )[[0, 2]]
        pybullet.changeDynamics(
            bodyUniqueId=identity,
            linkIndex=links[link_name],
            mass=mass*units.kilograms,
        )
        pybullet.changeDynamics(
            bodyUniqueId=identity,
            linkIndex=links[link_name],
            localInertiaDiagonal=np.array(inertias)*inertia_unit,
        )
    if morphology_links is not None:
        for link in morphology_links:
            if link not in links:
                links[link] = -1
                break
        for link in morphology_links:
            assert link in links, 'Link {} not in {}'.format(
                link,
                links,
            )
    return identity, links, joints
