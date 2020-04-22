#!/usr/bin/env python
""" Setup script """

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

DEBUG = False

setup(
    name='farms_bullet',
    version='0.1',
    author='farmsdev',
    author_email='biorob-farms@groupes.epfl.ch',
    description='FARMS package for running simulation with the Bullet engine',
    keywords='farms simulation bullet',
    packages=find_packages(),
    include_package_data=True,
    include_dirs=[np.get_include()],
    ext_modules=cythonize(
        [
            Extension(
                'farms_bullet.{}*'.format(folder.replace('/', '_') + '.' if folder else ''),
                sources=['farms_bullet/{}*.pyx'.format(folder + '/' if folder else '')],
                extra_compile_args=['-O3'],  # , '-fopenmp'
                extra_link_args=['-O3']  # , '-fopenmp'
            )
            for folder in [
                'data',
                'sensors'
            ]
        ],
        include_path=[np.get_include()],
        compiler_directives={
            'embedsignature': True,
            'cdivision': True,
            'language_level': 3,
            'infer_types': True,
            'profile': True,
            'wraparound': False,
            'boundscheck': DEBUG,
            'nonecheck': DEBUG,
            'initializedcheck': DEBUG,
            'overflowcheck': DEBUG,
        }
    ),
    zip_safe=False,
    # install_requires=[
    #     'cython',
    #     'numpy',
    #     'trimesh',
    #     'pybullet'
    # ],
)
