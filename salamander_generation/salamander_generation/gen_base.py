"""Generate base"""


import os

import xml.dom.minidom
from xml.etree import ElementTree as etree

from .model import (
    # create_directory,
    generate_model_options,
    ModelGenerationTemplates
)


class FrictionParameters:
    """FrictionParameters"""

    def __init__(self, body, feet):
        super(FrictionParameters, self).__init__()
        self._body = body
        self._feet = feet

    @property
    def feet(self):
        """Feet"""
        return self._feet

    @property
    def body(self):
        """Body"""
        return self._body


def apply_collisions_properties(root, friction_params):
    """Apply collisions properties"""
    for collision in root.iter('collision'):
        print("collision: {}".format(collision))
        print("tag: {} attribute: {}".format(collision.tag, collision.attrib))
        if "name" in collision.attrib:
            max_contacts = etree.Element("max_contacts")
            max_contacts.text = str(3)
            collision.append(max_contacts)
            surface = etree.Element("surface")
            collision.append(surface)
            # Friction
            friction = etree.Element("friction")
            surface.append(friction)
            # torsional = etree.Element("torsional")
            # friction.append(torsional)
            # coefficient = etree.Element("coefficient")
            # coefficient.text = str(friction_mu)
            # torsional.append(coefficient)
            ode = etree.Element("ode")
            friction.append(ode)
            mu = etree.Element("mu")
            mu.text = str(friction_params.feet)
            ode.append(mu)
            mu2 = etree.Element("mu2")
            mu2.text = str(friction_params.feet)
            ode.append(mu2)
            # Bounce
            bounce = etree.Element("bounce")
            surface.append(bounce)
            restitution_coefficient = etree.Element("restitution_coefficient")
            restitution_coefficient.text = str(0)
            bounce.append(restitution_coefficient)
            threshold = etree.Element("threshold")
            threshold.text = str(0)
            bounce.append(threshold)
            # Contact
            contact = etree.Element("contact")
            surface.append(contact)
            ode = etree.Element("ode")
            contact.append(ode)
            kp = etree.Element("kp")
            kp.text = str(1e6)
            ode.append(kp)
            kd = etree.Element("kd")
            kd.text = str(1e2)
            ode.append(kd)
            min_depth = etree.Element("min_depth")
            min_depth.text = str(1e2)
            ode.append(min_depth)


def correct_sdf_visuals_materials(root):
    """Correct materials from SDF visuals (DEPRECATED)

    Gazebo currently does not support obtaining materials from SDF directly. As
    the material incformation is encoded inside the COLLADA files, the material
    information obtained from the SDF file can be removed.

    """
    # Correct visuals materials (Gazebo/SDF does not fully support specular)
    for visual in root.iter('visual'):
        for stuff in visual.findall('material'):
            visual.remove(stuff)
    return root


def new_sdf_text(root):
    """Write new SDF file"""
    xml_string = etree.tostring(root).decode()
    xml_string = xml_string.replace("\n", "")
    while "  " in xml_string:
        xml_string = xml_string.replace("  ", "")
    xml_text = xml.dom.minidom.parseString(xml_string)
    return str(xml_text.toprettyxml(
        indent="  ",
        newl='\n'
        # encoding="UTF-8"
    ))


def write_new_sdf(root, path_sdf):
    """Write new SDF file (Deprecated)"""
    with open(path_sdf, "w+") as sdf_file:
        sdf_file.write(new_sdf_text(root))


def create_new_model(previous_model, new_model, friction):
    """Create new model from previous model"""
    home = os.path.expanduser("~")
    path_models = home + "/.gazebo/models/"
    path_model_previous = path_models+"{}/".format(previous_model)
    path_sdf_previous = path_model_previous+"{}.sdf".format(previous_model)
    path_model_new = path_models+"{}/".format(new_model)
    path_sdf_new = path_model_new+"{}.sdf".format(new_model)

    # SDF
    tree = etree.parse(path_sdf_previous)
    root = tree.getroot()

    # Collision properties
    apply_collisions_properties(root, friction)

    # Write to SDF
    sdf = new_sdf_text(root)

    # Generate package
    package = generate_model_options(
        name=new_model,
        base_model=previous_model
    )
    templates = ModelGenerationTemplates()
    packager = templates.render(package, sdf=sdf)
    packager.generate()


def generate_base():
    """Generate base models"""
    previous_model = "biorob_salamander_base"
    next_model = "biorob_salamander"
    friction = FrictionParameters(body=1e-3, feet=0.7)
    create_new_model(previous_model, next_model, friction)
