#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
diffrend setup
To develop: python setup.py develop --user
"""

from setuptools import setup, find_packages

# Find all packages
packages = find_packages()

# Add other file types to package
package_data = {}
for package in packages:
    package_data[package] = ['*.m', '*.c', '*.cpp', '*.h', '*.hpp', '*Makefile']

setup(name='diffrend',
      version='0.1',
      description='Differentiable Renderer',
      author='Fahim Mannan',
      author_email='fmannan@gmail.com',
      url='https://cim.mcgill.ca/~fmannan/diffrend',
      packages=packages,
      package_data=package_data,
      zip_safe=False,
      requires=['numpy', 'tensorflow', 'scipy', 'PyQt5', 'torch']
     )
