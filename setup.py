#!/usr/bin/env python
""" Setup script """

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np


setup(
    name="farms_bullet",
    version="0.1",
    author="farmsdev",
    author_email="jonathan.arreguitoneill@epfl.ch",
    description="FARMS package for running simulation with the Bullet engine",
    # license="BSD",
    keywords="farms simulation bullet",
    # url="",
    # packages=['farms_bullet'],
    packages=find_packages(),
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
    include_package_data=True
    ext_modules=cythonize([
        Extension(
            "farms_bullet.cy_controller",
            ["src/cy_controller.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=['-O3'],  # , '-fopenmp'
            extra_link_args=['-O3']  # , '-fopenmp'
        ),
        Extension(
            "farms_bullet.cy_controller_old",
            ["src/cy_controller_old.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=['-O3'],  # , '-fopenmp'
            extra_link_args=['-O3']  # , '-fopenmp'
        ),
        Extension(
            "farms_bullet.cy_controller_test",
            ["src/cy_controller_test.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=['-O3'],  # , '-fopenmp'
            extra_link_args=['-O3']  # , '-fopenmp'
        ),
    ])
)
