#!/usr/bin/env python
""" Setup script """

from setuptools import setup


setup(
    name="farms_bullet",
    version="0.1",
    author="farmsdev",
    author_email="jonathan.arreguitoneill@epfl.ch",
    description="FARMS package for running simulation with the Bullet engine",
    # license="BSD",
    keywords="farms simulation bullet",
    # url="",
    packages=['farms_bullet'],
    # long_description=read('README'),
    # classifiers=[
    #     "Development Status :: 3 - Alpha",
    #     "Topic :: Utilities",
    #     "License :: OSI Approved :: BSD License",
    # ],
    scripts=['scripts/farms_sim_salamander.py'],
    # package_data={'farms_bullet': [
    #     'farms_bullet/templates/*',
    #     'farms_bullet/config/*'
    # ]},
    # include_package_data=True
)
