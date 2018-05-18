#!/usr/bin/env python
import os
import shutil
import sys
from setuptools import setup, find_packages

VERSION = '0.0.1'

long_description = "this is my stuff for pytorch"

setup_info = dict(
    # Metadata
    name='pytorch_utils_plus',
    version=VERSION,
    author='psavine',
    author_email='psavine42@gmail.com',
    url='https://github.com/psavine42/pytorch_utils_plus',
    # url='https://github.com/pytorch/tnt/',
    description='an abstraction to train neural networks',
    long_description=long_description,
    license='BSD',

    # Package info
    packages=find_packages(exclude=('test', 'docs')),

    zip_safe=True,

    install_requires=[
        'torch',
        'six',
    ]
)

setup(**setup_info)