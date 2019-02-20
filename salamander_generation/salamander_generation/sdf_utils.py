"""SDF utils"""

import xml.dom.minidom
from xml.etree import ElementTree as etree


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
