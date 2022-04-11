#!/usr/bin/env python
"""Setup script"""

from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools import dist

dist.Distribution().fetch_build_eggs(['numpy'])
import numpy as np  # pylint: disable=wrong-import-position

dist.Distribution().fetch_build_eggs(['Cython>=0.15.1'])
from Cython.Build import cythonize  # pylint: disable=wrong-import-position
from Cython.Compiler import Options  # pylint: disable=wrong-import-position

dist.Distribution().fetch_build_eggs(['farms_core'])
from farms_core import get_include_paths  # pylint: disable=wrong-import-position


# Cython options
DEBUG = False
Options.docstrings = True
Options.embed_pos_in_docstring = False
Options.generate_cleanup_code = False
Options.clear_to_none = True
Options.annotate = False
Options.fast_fail = False
Options.warning_errors = False
Options.error_on_unknown_names = True
Options.error_on_uninitialized = True
Options.convert_range = True
Options.cache_builtins = True
Options.gcc_branch_hints = True
Options.lookup_module_cpdef = False
Options.embed = None
Options.cimport_from_pyx = False
Options.buffer_max_dims = 8
Options.closure_freelist_size = 8


setup(
    name='farms_mujoco',
    version='0.1',
    author='farmsdev',
    author_email='biorob-farms@groupes.epfl.ch',
    description='FARMS package for running simulations with MuJoCo',
    keywords='farms simulation mujoco',
    packages=find_packages(),
    package_dir={'farms_mujoco': 'farms_mujoco'},
    package_data={'farms_mujoco': [
        f'{folder}/*.pxd'
        for folder in ['sensors', 'swimming']
    ]},
    include_package_data=True,
    include_dirs=[np.get_include()] + get_include_paths(),
    ext_modules=cythonize(
        [
            Extension(
                f'farms_mujoco.{folder}.*',
                sources=[f'farms_mujoco/{folder}/*.pyx'],
                extra_compile_args=['-O3'],  # , '-fopenmp'
                extra_link_args=['-O3']  # , '-fopenmp'
            )
            for folder in ['sensors', 'swimming']
        ],
        include_path=[np.get_include()] + get_include_paths(),
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
    install_requires=[
        'farms_core',
        'cython',
        'numpy',
        'scipy',
        'tqdm'
        'trimesh',
        'dm_control',
        'imageio',
    ],
)
