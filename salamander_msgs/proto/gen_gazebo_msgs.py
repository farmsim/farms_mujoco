""" Gazebo protobuf messages generation for Python """

import os
from subprocess import call


def main():
    """ Main """
    include_dir = "/usr/include/gazebo-9/gazebo/msgs/proto"
    python_out = "./../salamander_msgs/gazebo"
    command = "protoc -I={} --python_out={} {}"
    for filename in os.listdir(include_dir):
        cmd = command.format(include_dir, python_out, include_dir+"/"+filename)
        print(cmd)
        call(cmd, shell=True)


if __name__ == '__main__':
    main()
