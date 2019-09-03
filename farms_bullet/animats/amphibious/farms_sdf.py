"""Farms SDF"""

import xml.etree.cElementTree as ET
import xml.dom.minidom

from ...simulations.simulation_options import Options


class ModelSDF(Options):
    """Farms SDF"""

    def __init__(self, name="animat"):
        super(ModelSDF, self).__init__()
        self.name = name
        self.pose = "0 0 0 0 0 0"
        self.links = [
            Link(name="base_link")
        ]
        self.joints = [
            Joint(
                name="joint_0",
                joint_type="revolute",
                parent=self.links[0],
                child=self.links[0]
            )
        ]
        self.xml()

    def xml(self):
        """xml"""
        sdf = ET.Element("sdf", version="1.5")
        model = ET.SubElement(sdf, "model", name=self.name)

        pose = ET.SubElement(model, "pose")
        pose.text = self.pose
        for link in self.links:
            link.xml(model)
        for joint in self.joints:
            joint.xml(model)

        tree = ET.ElementTree(sdf)
        xml_str = ET.tostring(
            sdf,
            encoding='utf8',
            method='xml'
        ).decode('utf8')
        # dom = xml.dom.minidom.parse(xml_fname)
        dom = xml.dom.minidom.parseString(xml_str)
        pretty_xml_as_string = dom.toprettyxml(indent=2*" ")
        print(pretty_xml_as_string)
        # tree.write("filename.sdf")


class Link(Options):
    """Link"""

    def __init__(self, name="base_link"):
        super(Link, self).__init__()
        self.name = name
        self.inertial = Inertial()
        self.collision = Collision()
        self.visual = Visual()

    def xml(self, model):
        """xml"""
        link = ET.SubElement(model, "link")
        self.inertial.xml(link)
        self.collision.xml(link)
        self.visual.xml(link)


class Inertial(Options):
    """Inertial"""

    def __init__(self, mass=1, inertias=None):
        super(Inertial, self).__init__()
        self.mass = str(mass)
        self.inertias = inertias
        if self.inertias is None:
            self.inertias = ["1"]*6

    def xml(self, link):
        """xml"""
        inertial = ET.SubElement(link, "inertial")
        mass = ET.SubElement(inertial, "mass")
        mass.text = str(self.mass)
        inertia = ET.SubElement(inertial, "inertia")
        inertias = [
            ET.SubElement(inertia, name)
            for name in ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]
        ]
        for i, inertia in enumerate(inertias):
            inertia.text = self.inertias[i]


class Collision(Options):
    """Collision"""

    def __init__(self, name="base_link"):
        super(Collision, self).__init__()
        self.name = "{}_collision".format(name)
        self.pose = "0 0 0 0 0 0"
        self.geometry = Box()

    def xml(self, link):
        """xml"""
        collision = ET.SubElement(
            link,
            "collision",
            name=self.name
        )
        pose = ET.SubElement(collision, "pose")
        pose.text = self.pose
        self.geometry.xml(collision)


class Visual(Options):
    """Visual"""

    def __init__(self, name="base_link"):
        super(Visual, self).__init__()
        self.name = "{}_visual".format(name)
        self.pose = "0 0 0 0 0 0"
        self.geometry = Box()

    def xml(self, link):
        """xml"""
        visual = ET.SubElement(
            link,
            "visual",
            name=self.name
        )
        pose = ET.SubElement(visual, "pose")
        pose.text = self.pose
        self.geometry.xml(visual)


class Box(Options):
    """Box"""

    def __init__(self):
        super(Box, self).__init__()
        self.size = "0 0 0"

    def xml(self, parent):
        """xml"""
        geometry = ET.SubElement(parent, "geometry")
        box = ET.SubElement(geometry, "box")
        size = ET.SubElement(box, "size")
        size.text = self.size


class Joint(Options):
    """Joint"""

    def __init__(self, name, joint_type, parent, child):
        super(Joint, self).__init__()
        self.name = name
        self.type = joint_type
        self.parent = parent.name
        self.child = child.name
        self.pose = "0 0 0 0 0 0"
        self.axis = Axis()

    def xml(self, model):
        """xml"""
        joint = ET.SubElement(model, "joint", name=self.name, type=self.type)
        parent = ET.SubElement(joint, "parent")
        parent.text = self.parent
        child = ET.SubElement(joint, "child")
        child.text = self.child
        pose = ET.SubElement(joint, "pose")
        pose.text = self.pose
        self.axis.xml(joint)


class Axis(Options):
    """Axis"""

    def __init__(self):
        super(Axis, self).__init__()
        self.initial_position = "0"
        self.xyz = "0 0 1"

    def xml(self, joint):
        """xml"""
        axis = ET.SubElement(joint, "axis")
        xyz = ET.SubElement(axis, "xyz")
        xyz.text = self.xyz
