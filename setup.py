#!/usr/bin/env python3

from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


# Get the program version from another file.
__version__ = '0.0.0'
exec(open('deepbinner/version.py').read())


setup(name='Deepbinner',
      version=__version__,
      description='Deepbinner: a deep convolutional neural network barcode demultiplexer for '
                  'Oxford Nanopore reads',
      long_description=readme(),
      url='https://github.com/rrwick/Deepbinner',
      author='Ryan Wick',
      author_email='rrwick@gmail.com',
      entry_points={"console_scripts": ['deepbinner = deepbinner.deepbinner:main']},
      include_package_data=True,
      zip_safe=False)
