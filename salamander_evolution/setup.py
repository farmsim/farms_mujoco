#!/usr/bin/env python
""" Setup script """

from setuptools import setup


setup(
    name="salamander_evolution",
    version="0.1",
    author="Jonathan Arreguit",
    author_email="jonathan.arreguitoneill@epfl.ch",
    description="Salamander evolution",
    # license="BSD",
    keywords="salamander evolution gazebo",
    # url="",
    packages=['salamander_evolution'],
    # long_description=read('README'),
    # classifiers=[
    #     "Development Status :: 3 - Alpha",
    #     "Topic :: Utilities",
    #     "License :: OSI Approved :: BSD License",
    # ],
    scripts=['scripts/salamander_test_evolution.py'],
    # package_data={'salamander_evolution': [
    #     'salamander_evolution/templates/*',
    #     'salamander_evolution/config/*'
    # ]},
    # include_package_data=True
)
