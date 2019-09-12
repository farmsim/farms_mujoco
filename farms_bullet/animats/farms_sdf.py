"""Farms SDF"""

import xml.etree.cElementTree as ET
import xml.dom.minidom

import numpy as np
import trimesh as tri

from ..simulations.simulation_options import Options


class ModelSDF(Options):
    """Farms SDF"""

    def __init__(self, name, pose, units, **kwargs):
        super(ModelSDF, self).__init__()
        self.name = name
        self.pose = pose
        self.units = units
        self.links = kwargs.pop("links", [])
        self.joints = kwargs.pop("joints", [])

    def xml(self):
        """xml"""
        sdf = ET.Element("sdf", version="1.5")
        model = ET.SubElement(sdf, "model", name=self.name)
        pose = ET.SubElement(model, "pose")
        pose.text = " ".join([
            str(element*(self.units.meters if i < 3 else 1))
            for i, element in enumerate(self.pose)
        ])
        for link in self.links:
            link.xml(model)
        for joint in self.joints:
            joint.xml(model)
        return sdf

    def xml_str(self):
        """xml string"""
        sdf = self.xml()
        xml_str = ET.tostring(
            sdf,
            encoding='utf8',
            method='xml'
        ).decode('utf8')
        # dom = xml.dom.minidom.parse(xml_fname)
        dom = xml.dom.minidom.parseString(xml_str)
        return dom.toprettyxml(indent=2*" ")

    def write(self, filename="animat.sdf"):
        """Write SDF to file"""
        # ET.ElementTree(self.xml()).write(filename)
        with open(filename, "w+") as sdf_file:
            sdf_file.write(self.xml_str())


class Link(Options):
    """Link"""

    def __init__(self, name, pose, inertial, collision, visual, units):
        super(Link, self).__init__()
        self.name = name
        self.pose = pose
        self.inertial = inertial
        self.collision = collision
        self.visual = visual
        self.units = units

    @classmethod
    def box(cls, name, pose, units, **kwargs):
        """Box"""
        visual_kwargs = {}
        if "color" in kwargs:
            visual_kwargs["color"] = kwargs.pop("color", None)
        # inertial_pose = kwargs.pop("inertial_pose", np.zeros(6))
        shape_pose = kwargs.pop("shape_pose", np.zeros(6))
        return cls(
            name,
            pose=pose,
            inertial=Inertial.box(
                pose=shape_pose,
                units=units,
                **kwargs
            ),
            collision=Collision.box(
                name,
                pose=shape_pose,
                units=units,
                **kwargs
            ),
            visual=Visual.box(
                name,
                pose=shape_pose,
                units=units,
                **visual_kwargs,
                **kwargs
            ),
            units=units
        )

    @classmethod
    def sphere(cls, name, pose, units, **kwargs):
        """Sphere"""
        visual_kwargs = {}
        if "color" in kwargs:
            visual_kwargs["color"] = kwargs.pop("color", None)
        # inertial_pose = kwargs.pop("inertial_pose", np.zeros(6))
        shape_pose = kwargs.pop("shape_pose", np.zeros(6))
        return cls(
            name,
            pose=pose,
            inertial=Inertial.sphere(
                pose=shape_pose,
                units=units,
                **kwargs,
            ),
            collision=Collision.sphere(
                name,
                pose=shape_pose,
                units=units,
                **kwargs
            ),
            visual=Visual.sphere(
                name,
                pose=shape_pose,
                units=units,
                **visual_kwargs,
                **kwargs,
            ),
            units=units
        )

    @classmethod
    def capsule(cls, name, pose, units, **kwargs):
        """Capsule"""
        visual_kwargs = {}
        if "color" in kwargs:
            visual_kwargs["color"] = kwargs.pop("color", None)
        # inertial_pose = kwargs.pop("inertial_pose", np.zeros(6))
        shape_pose = kwargs.pop("shape_pose", np.zeros(6))
        return cls(
            name,
            pose=pose,
            inertial=Inertial.capsule(
                pose=shape_pose,
                units=units,
                **kwargs
            ),
            collision=Collision.capsule(
                name,
                pose=shape_pose,
                units=units,
                **kwargs
            ),
            visual=Visual.capsule(
                name,
                pose=shape_pose,
                units=units,
                **visual_kwargs,
                **kwargs
            ),
            units=units
        )

    @classmethod
    def from_mesh(cls, name, mesh, pose, scale, units, **kwargs):
        """From mesh"""
        visual_kwargs = {}
        if "color" in kwargs:
            visual_kwargs["color"] = kwargs.pop("color", None)
        # inertial_pose = kwargs.pop("inertial_pose", np.zeros(6))
        shape_pose = kwargs.pop("shape_pose", np.zeros(6))
        return cls(
            name,
            pose=pose,
            inertial=Inertial.from_mesh(
                mesh,
                pose=shape_pose,
                scale=scale,
                units=units
            ),
            collision=Collision.from_mesh(
                name,
                mesh,
                pose=shape_pose,
                scale=scale,
                units=units
            ),
            visual=Visual.from_mesh(
                name,
                mesh,
                pose=shape_pose,
                scale=scale,
                units=units,
                **visual_kwargs
            ),
            units=units
        )

    def xml(self, model):
        """xml"""
        link = ET.SubElement(model, "link", name=self.name)
        pose = ET.SubElement(link, "pose")
        pose.text = " ".join([
            str(element*(self.units.meters if i < 3 else 1))
            for i, element in enumerate(self.pose)
        ])
        self.inertial.xml(link)
        self.collision.xml(link)
        self.visual.xml(link)


class Inertial(Options):
    """Inertial"""

    def __init__(self, pose, mass, inertias, units):
        super(Inertial, self).__init__()
        self.mass = str(mass)
        self.inertias = inertias
        self.units = units
        self.pose = pose

    @classmethod
    def box(cls, size, pose, units, **kwargs):
        """Box"""
        density = kwargs.pop("density", 1000)
        volume = size[0]*size[1]*size[2]
        mass = volume*density
        return cls(
            pose=np.asarray(pose),
            mass=mass,
            inertias=[
                1/12*mass*(size[1]**2 + size[2]**2),
                0,
                0,
                1/12*mass*(size[0]**2 + size[2]**2),
                0,
                1/12*mass*(size[0]**2 + size[1]**2)
            ],
            units=units
        )

    @classmethod
    def sphere(cls, radius, pose, units, **kwargs):
        """Sphere"""
        density = kwargs.pop("density", 1000)
        volume = 4/3*np.pi*radius**3
        mass = volume*density
        return cls(
            pose=np.asarray(pose),
            mass=mass,
            inertias=[
                2/5*mass*radius**2,
                0,
                0,
                2/5*mass*radius**2,
                0,
                2/5*mass*radius**2
            ],
            units=units
        )

    @classmethod
    def capsule(cls, length, radius, pose, units, **kwargs):
        """Capsule"""
        density = kwargs.pop("density", 1000)
        volume_sphere = 4/3*np.pi*radius**3
        volume_cylinder = np.pi*radius**2*length
        volume = volume_sphere + volume_cylinder
        mass = volume*density
        return cls(
            pose=np.asarray(pose),
            mass=mass,
            # TODO: This is Cylinder inertia!!
            inertias=[
                1/12*mass*(3*radius**2 + length**2),
                0,
                0,
                1/12*mass*(3*radius**2 + length**2),
                0,
                1/2*mass*(radius**2)
            ],
            units=units
        )

    @classmethod
    def from_mesh(cls, mesh, scale, pose, units, **kwargs):
        """From mesh"""
        density = kwargs.pop("density", 1000)
        _mesh = tri.load_mesh(mesh)
        volume = scale*_mesh.volume
        inertia = scale**2*_mesh.moment_inertia
        return cls(
            pose=np.concatenate([
                scale*(_mesh.center_mass+np.asarray(pose[:3])),
                pose[3:]
            ]),
            mass=volume*density,
            inertias=[
                inertia[0, 0],
                inertia[0, 1],
                inertia[0, 2],
                inertia[1, 1],
                inertia[1, 2],
                inertia[2, 2]
            ],
            units=units
        )

    def xml(self, link):
        """xml"""
        inertial = ET.SubElement(link, "inertial")
        pose = ET.SubElement(inertial, "pose")
        pose.text = " ".join([
            str(element*(self.units.meters if i < 3 else 1))
            for i, element in enumerate(self.pose)
        ])
        mass = ET.SubElement(inertial, "mass")
        mass.text = str(self.mass*self.units.kilograms)
        inertia = ET.SubElement(inertial, "inertia")
        inertias = [
            ET.SubElement(inertia, name)
            for name in ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]
        ]
        for i, inertia in enumerate(inertias):
            inertia.text = str(
                self.inertias[i]*self.units.meters**5
            )


class Shape(Options):
    """Shape"""

    def __init__(self, name, geometry, suffix, units, pose=np.zeros(6)):
        super(Shape, self).__init__()
        self.name = "{}_{}".format(name, suffix)
        self.geometry = geometry
        self.suffix = suffix
        self.pose = pose
        self.units = units

    @classmethod
    def box(cls, name, size, units, **kwargs):
        """Box"""
        return cls(
            name=name,
            geometry=Box(size, units),
            units=units,
            **kwargs
        )

    @classmethod
    def sphere(cls, name, radius, units, **kwargs):
        """Box"""
        return cls(
            name=name,
            geometry=Sphere(radius, units),
            units=units,
            **kwargs
        )

    @classmethod
    def capsule(cls, name, length, radius, units, **kwargs):
        """Box"""
        return cls(
            name=name,
            geometry=Capsule(length, radius, units),
            units=units,
            **kwargs
        )

    @classmethod
    def from_mesh(cls, name, mesh, scale, units, **kwargs):
        """From mesh"""
        return cls(
            name=name,
            geometry=Mesh(mesh, scale, units),
            units=units,
            **kwargs
        )

    def xml(self, link):
        """xml"""
        shape = ET.SubElement(
            link,
            self.suffix,
            name=self.name
        )
        pose = ET.SubElement(shape, "pose")
        pose.text = " ".join([
            str(element*(self.units.meters if i < 3 else 1))
            for i, element in enumerate(self.pose)
        ])
        self.geometry.xml(shape)
        return shape


class Collision(Shape):
    """Collision"""

    SUFFIX = "collision"

    def __init__(self, name, **kwargs):
        super(Collision, self).__init__(name=name, suffix=self.SUFFIX, **kwargs)


class Visual(Shape):
    """Visual"""

    SUFFIX = "visual"

    def __init__(self, name, **kwargs):
        self.color = kwargs.pop("color", None)
        self.ambient = self.color
        self.diffuse = self.color
        self.specular = self.color
        self.emissive = self.color
        super(Visual, self).__init__(name=name, suffix=self.SUFFIX, **kwargs)

    def xml(self, link):
        """xml"""
        shape = super(Visual, self).xml(link)
        material = ET.SubElement(shape, "material")
        # script = ET.SubElement(material, "script")
        # uri = ET.SubElement(script, "uri")
        # uri.text = "skin.material"
        # name = ET.SubElement(script, "name")
        # name.text = "Skin"
        if self.color is not None:
            # color = ET.SubElement(material, "color")
            # color.text = " ".join([str(element) for element in self.color])
            ambient = ET.SubElement(material, "ambient")
            ambient.text = " ".join([str(element) for element in self.ambient])
            diffuse = ET.SubElement(material, "diffuse")
            diffuse.text = " ".join([str(element) for element in self.diffuse])
            specular = ET.SubElement(material, "specular")
            specular.text = " ".join([str(element) for element in self.specular])
            emissive = ET.SubElement(material, "emissive")
            emissive.text = " ".join([str(element) for element in self.emissive])


class Box(Options):
    """Box"""

    def __init__(self, size, units):
        super(Box, self).__init__()
        self.size = size
        self.units = units

    def xml(self, parent):
        """xml"""
        geometry = ET.SubElement(parent, "geometry")
        box = ET.SubElement(geometry, "box")
        size = ET.SubElement(box, "size")
        size.text = " ".join([
            str(element*self.units.meters)
            for element in self.size
        ])


class Sphere(Options):
    """Sphere"""

    def __init__(self, radius, units):
        super(Sphere, self).__init__()
        self.radius = radius
        self.units = units

    def xml(self, parent):
        """xml"""
        geometry = ET.SubElement(parent, "geometry")
        sphere = ET.SubElement(geometry, "sphere")
        radius = ET.SubElement(sphere, "radius")
        radius.text = str(self.radius*self.units.meters)


class Capsule(Options):
    """Capsule"""

    def __init__(self, length, radius, units):
        super(Capsule, self).__init__()
        self.length = length
        self.radius = radius
        self.units = units

    def xml(self, parent):
        """xml"""
        geometry = ET.SubElement(parent, "geometry")
        capsule = ET.SubElement(geometry, "capsule")
        length = ET.SubElement(capsule, "length")
        length.text = str(self.length*self.units.meters)
        radius = ET.SubElement(capsule, "radius")
        radius.text = str(self.radius*self.units.meters)


class Mesh(Options):
    """Mesh"""

    def __init__(self, uri, scale, units):
        super(Mesh, self).__init__()
        self.uri = uri
        self.scale = scale
        self.units = units

    def xml(self, parent):
        """xml"""
        geometry = ET.SubElement(parent, "geometry")
        mesh = ET.SubElement(geometry, "mesh")
        uri = ET.SubElement(mesh, "uri")
        uri.text = self.uri
        scale = ET.SubElement(mesh, "scale")
        scale.text = " ".join([str(self.scale*self.units.meters)]*3)


class Joint(Options):
    """Joint"""

    def __init__(self, name, joint_type, parent, child, **kwargs):
        super(Joint, self).__init__()
        self.name = name
        self.type = joint_type
        self.parent = parent.name
        self.child = child.name
        self.pose = np.zeros(6)
        self.axis = kwargs.pop("axis", None)
        if self.axis is not None:
            self.axis = Axis(xyz=self.axis, **kwargs)

    def xml(self, model):
        """xml"""
        joint = ET.SubElement(model, "joint", name=self.name, type=self.type)
        parent = ET.SubElement(joint, "parent")
        parent.text = self.parent
        child = ET.SubElement(joint, "child")
        child.text = self.child
        pose = ET.SubElement(joint, "pose")
        pose.text = " ".join([str(element) for element in self.pose])
        if self.axis:
            self.axis.xml(joint)


class Axis(Options):
    """Axis"""

    def __init__(self, **kwargs):
        super(Axis, self).__init__()
        self.initial_position = kwargs.pop("initial_position", None)
        self.xyz = kwargs.pop("xyz", [0, 0, 0])
        self.limits = kwargs.pop("limits", None)

    def xml(self, joint):
        """xml"""
        axis = ET.SubElement(joint, "axis")
        if self.initial_position:
            initial_position = ET.SubElement(axis, "initial_position")
            initial_position.text = " ".join(self.initial_position)
        xyz = ET.SubElement(axis, "xyz")
        xyz.text = " ".join([str(element) for element in self.xyz])
        if self.limits is not None:
            limit = ET.SubElement(axis, "limit")
            lower = ET.SubElement(limit, "lower")
            lower.text = str(self.limits[0])
            upper = ET.SubElement(limit, "upper")
            upper.text = str(self.limits[1])
            effort = ET.SubElement(limit, "effort")
            effort.text = str(self.limits[2])
            velocity = ET.SubElement(limit, "velocity")
            velocity.text = str(self.limits[3])
