#!/usr/bin/env python
""" Setup script """

from setuptools import setup

setup(
    name="salamander_sweep",
    version="0.1",
    author="Jonathan Arreguit",
    author_email="jonathan.arreguitoneill@epfl.ch",
    description="Salamander sweep",
    # license="BSD",
    keywords="salamander sweep",
    # url="",
    packages=['salamander_sweep'],
    scripts=[
        "scripts/salamander_run_sweep.py",
        "scripts/salamander_test_sweep.py"
    ]
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
