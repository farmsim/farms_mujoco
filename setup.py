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
    # license='BSD',
    keywords='farms simulation bullet',
    # url='',
    # packages=['farms_bullet'],
    packages=find_packages(),
    # long_description=read('README'),
    # classifiers=[
    #     'Development Status :: 3 - Alpha',
    #     'Topic :: Utilities',
    #     'License :: OSI Approved :: BSD License',
    # ],
    scripts=[
        # 'scripts/farms_salamander.py',
        # 'scripts/farms_snake.py',
        # 'scripts/farms_centipede.py',
        # 'scripts/farms_polypterus.py',
        # 'scripts/farms_quadruped.py'
    ],
    # package_data={'farms_bullet': [
    #     'farms_bullet/templates/*',
    #     'farms_bullet/config/*'
    # ]},
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
                # 'animats/data',
                # 'controllers',
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
