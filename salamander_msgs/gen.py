#!/usr/bin/env python
""" Gazebo protobuf messages generation for Python """

import os
from subprocess import call
from shutil import copyfile
import re


def correct_file_callback_py(pat):
    """ Correct file callback """
    return "from . import {} as {}".format(pat.group(1), pat.group(2))


def correct_file_py(filename, regex):
    """ Correct_file """
    print("  Correcting {}".format(filename))
    with open(filename, "r") as _file:
        data = _file.read()
    data_new = re.sub(regex, correct_file_callback_py, data)
    with open(filename, "w") as _file:
        _file.write(data_new)


def correct_file_cpp(filename, pattern, replacement):
    """ Correct_file """
    print("  Correcting {}".format(filename))
    with open(filename, "r") as _file:
        data = _file.read()
    data_new = data.replace(pattern, replacement)
    with open(filename, "w") as _file:
        _file.write(data_new)


def copy_proto_files(gazebo_dir, include_dir):
    """ Copy proto files from Gazebo """
    gazebo_files = os.listdir(gazebo_dir)
    for filename in gazebo_files:
        copyfile(gazebo_dir+"/"+filename, include_dir+"/"+filename)
    return gazebo_files


def gen_salamander_msgs(files, include_dir, output, language, extension):
    """ Generate salamander msgs """
    command = "protoc --proto_path={} --proto_path={} --{}_out={} {}"
    for filename in files:
        cmd = command.format(
            include_dir,
            "./proto",
            language,
            output,
            filename
        )
        print(cmd)
        call(cmd, shell=True)
        # Correct files
        filename_out = (
            output + "/"
            + filename.split("/")[-1].split(".")[0]
            + extension
        )
        if language == "cpp":
            with open(filename_out, "r") as _file:
                data = _file.read()
            for search in re.finditer(r'\#include\ \"(\w+)\.pb\.h\"', data):
                print("Search FOUND: {}".format(search.group(0)))
                if "./proto/" + search.group(1) + ".proto" not in files:
                    print("  Correcting {}, gazebo message deteted".format(
                        search.group(0)
                    ))
                    pattern = '#include "{}.pb.h"'.format(search.group(1))
                    replacement = '#include "gazebo/msgs/{}.pb.h"'.format(
                        search.group(1)
                    )
                    print("  Replacing {} with {}".format(
                        pattern,
                        replacement
                    ))
                    correct_file_cpp(
                        filename_out,
                        pattern=pattern,
                        replacement=replacement
                    )
                else:
                    break
        elif language == "python":
            correct_file_py(
                filename_out,
                regex=r"import\ (\w+\_pb2)\ as\ (\w+\_\_pb2)"
            )
        else:
            raise Exception("Unrecognised language '{}'".format(language))


def gen_all():
    """ Generate cpp and python messages """
    print("Generating salamander_msgs protobuf library")
    gazebo_dir = "/usr/include/gazebo-9/gazebo/msgs/proto"
    include_dir = "./proto/gazebo"
    gazebo_files = copy_proto_files(gazebo_dir, include_dir)
    files = [
        "./proto/salamander_kinematics.proto",
        "./proto/salamander_links.proto",
        "./proto/salamander_joints.proto",
        "./proto/salamander_control.proto"
    ]
    gen_salamander_msgs(
        files,
        include_dir,
        output="./salamander_msgs_cpp",
        language="cpp",
        extension=".pb.h"
    )
    files = files + [include_dir+"/"+filename for filename in gazebo_files]
    gen_salamander_msgs(
        files,
        include_dir,
        output="./salamander_msgs",
        language="python",
        extension="_pb2.py"
    )


if __name__ == '__main__':
    gen_all()
