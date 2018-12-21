#!/usr/bin/env python
""" Setup script """

from subprocess import call
from setuptools import setup

print("Generating salamander_msgs protobuf library")
call("protoc --proto_path={} --python_out={} {}".format(
    "./proto",
    "./salamander_msgs",
    "./proto/log_kinematics.proto"
), shell=True)

setup(
    name="salamander_msgs",
    version="0.1",
    author="Jonathan Arreguit",
    author_email="jonathan.arreguitoneill@epfl.ch",
    description="Salamander messages",
    # license="BSD",
    keywords="salamander messages",
    # url="",
    packages=['salamander_msgs'],
    # long_description=read('README'),
    # classifiers=[
    #     "Development Status :: 3 - Alpha",
    #     "Topic :: Utilities",
    #     "License :: OSI Approved :: BSD License",
    # ],
    # package_data={'salamander_messages': [
    #     'salamander_generation/templates/*',
    #     'salamander_generation/config/*'
    # ]},
    # include_package_data=True
)
