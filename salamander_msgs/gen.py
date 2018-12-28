#!/usr/bin/env python
""" Gazebo protobuf messages generation for Python """

import os
from subprocess import call
from shutil import copyfile
import re


def correct_file_callback_py(pat):
    """ Correct file callback """
    return "from . import {} as {}".format(pat.group(1), pat.group(2))


def correct_file_callback_cpp(pat):
    """ Correct file callback """
    return '#include "gazebo/msgs/{}"'.format(pat.group(1))


def correct_file(filename, regex, callback):
    """ Correct_file """
    print("Correcting {}".format(filename))
    with open(filename, "r") as _file:
        data = _file.read()
    data_new = re.sub(regex, callback, data)
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
            correct_file(
                filename_out,
                regex=r'\#include\ \"(\w+\.pb\.h)\"',
                callback=correct_file_callback_cpp
            )
        elif language == "python":
            correct_file(
                filename_out,
                regex=r"import\ (\w+\_pb2)\ as\ (\w+\_\_pb2)",
                callback=correct_file_callback_py
            )
        else:
            raise Exception("Unrecognised language '{}'".format(language))


def gen_all():
    """ Generate cpp and python messages """
    print("Generating salamander_msgs protobuf library")
    gazebo_dir = "/usr/include/gazebo-9/gazebo/msgs/proto"
    include_dir = "./proto/gazebo"
    gazebo_files = copy_proto_files(gazebo_dir, include_dir)
    files = ["./proto/log_kinematics.proto"]
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


# def gen_salamander_msgs_py():
#     """ Generate salamander msgs - python """
#     print("Generating salamander_msgs protobuf library")
#     gazebo_dir = "/usr/include/gazebo-9/gazebo/msgs/proto"
#     include_dir = "./proto/gazebo"
#     # Copy files
#     for filename in os.listdir(gazebo_dir):
#         copyfile(gazebo_dir+"/"+filename, include_dir+"/"+filename)
#     python_out = "./salamander_msgs"
#     command = "protoc --proto_path={} --proto_path={} --python_out={} {}"
#     files = ["./proto/log_kinematics.proto"] + [
#         include_dir+"/"+filename
#         for filename in os.listdir(include_dir)
#     ]
#     for filename in files:
#         cmd = command.format(
#             include_dir,
#             "./proto",
#             python_out,
#             filename
#         )
#         print(cmd)
#         call(cmd, shell=True)
#         # Correct files
#         pyfile = (
#             python_out + "/"
#             + filename.split("/")[-1].split(".")[0]
#             + "_pb2.py"
#         )
#         correct_file(
#             pyfile,
#             regex=r"import\ (\w+\_pb2)\ as\ (\w+\_\_pb2)",
#             callback=correct_file_callback_py
#         )


if __name__ == '__main__':
    gen_all()
