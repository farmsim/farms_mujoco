#!/usr/bin/env python
""" Setup script """

from setuptools import setup


setup(
    name="salamander_generation",
    version="0.1",
    author="Jonathan Arreguit",
    author_email="jonathan.arreguitoneill@epfl.ch",
    description="Salamander generation",
    # license="BSD",
    keywords="salamander model generation gazebo",
    # url="",
    packages=['salamander_generation'],
    # long_description=read('README'),
    # classifiers=[
    #     "Development Status :: 3 - Alpha",
    #     "Topic :: Utilities",
    #     "License :: OSI Approved :: BSD License",
    # ],
    package_data={'salamander_generation': [
        'salamander_generation/templates/*',
        'salamander_generation/config/*'
    ]},
    include_package_data=True
)
