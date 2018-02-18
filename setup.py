#!/usr/bin/env python3

import os
import sys
import subprocess
from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


__version__ = '0.1.0'


setup(name='CNN Demultiplexer',
      version=__version__,
      description='CNN barcode demultiplexer for Oxford Nanopore reads',
      long_description=readme(),
      url='https://github.com/rrwick/CNN-Demultiplexer',
      author='Ryan Wick',
      author_email='',
      entry_points={"console_scripts": ['cnn_demultiplexer = cnn_demultiplexer.cnn_demultiplexer:main']},
      include_package_data=True,
      zip_safe=False)
