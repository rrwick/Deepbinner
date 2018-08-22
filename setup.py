#!/usr/bin/env python3

import os
import shlex
import shutil
import subprocess
import sys
from setuptools import setup
from setuptools.command.install import install


def readme():
    with open('README.md') as f:
        return f.read()


# Get the program version from another file.
__version__ = '0.0.0'
exec(open('deepbinner/version.py').read())


class DeepbinnerInstall(install):
    """
    The install process copies the C++ shared library to the install location.
    """
    user_options = install.user_options + [
        ('makeargs=', None, 'Arguments to be given to make')
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.makeargs = None

    if __name__ == '__main__':
        def run(self):
            # Make sure we have permission to write the files.
            if os.path.isdir(self.install_lib) and not os.access(self.install_lib, os.W_OK):
                sys.exit('Error: no write permission for ' + self.install_lib + '  ' +
                         'Perhaps you need to use sudo?')
            if os.path.isdir(self.install_scripts) and not os.access(self.install_scripts, os.W_OK):
                sys.exit('Error: no write permission for ' + self.install_scripts + '  ' +
                         'Perhaps you need to use sudo?')

            # Clean up any previous Deepbinner compilation.
            clean_cmd = ['make', 'distclean']
            self.execute(lambda: subprocess.call(clean_cmd), [],
                         'Cleaning previous compilation: ' + ' '.join(clean_cmd))

            # Build Deepbinner's C++ code.
            make_cmd = ['make']
            if self.makeargs:
                make_cmd += shlex.split(self.makeargs)
            self.execute(lambda: subprocess.call(make_cmd), [],
                         'Compiling Deepbinner: ' + ' '.join(make_cmd))
            cpp_code = os.path.join('deepbinner', 'dtw', 'dtw.so')
            compile_success = os.path.isfile(cpp_code)

            install.run(self)

            if compile_success:
                # Copy the C++ library to the installation location.
                try:
                    os.makedirs(os.path.join(self.install_lib, 'deepbinner', 'dtw'))
                except FileExistsError:
                    pass
                shutil.copyfile(cpp_code,
                                os.path.join(self.install_lib, 'deepbinner', 'dtw', 'dtw.so'))
            else:
                print('\nWarning: there was a problem building Deepbinner\'s C++ code (only a ')
                print('           cause for concern if you were going to train a custom ')
                print('           Deepbinner neural network)')

            # Copy the pre-trained models to the install location.
            try:
                os.makedirs(os.path.join(self.install_lib, 'deepbinner', 'models'))
            except FileExistsError:
                pass
            print()
            for model in ['EXP-NBD103_read_starts', 'EXP-NBD103_read_ends',
                          'SQK-RBK004_read_starts']:
                install_model_dir = os.path.join(self.install_lib, 'deepbinner', 'models')
                print('Copying {} to {}'.format(model, install_model_dir))
                model_file = os.path.join('models', model)
                shutil.copy(model_file, os.path.join(self.install_lib, 'deepbinner', 'models'))

            print('\nDeepbinner is installed!\n')


setup(name='Deepbinner',
      version=__version__,
      description='Deepbinner: a deep convolutional neural network barcode demultiplexer for '
                  'Oxford Nanopore reads',
      long_description=readme(),
      url='https://github.com/rrwick/Deepbinner',
      author='Ryan Wick',
      author_email='rrwick@gmail.com',
      license='GPLv3',
      packages=['deepbinner'],
      entry_points={"console_scripts": ['deepbinner = deepbinner.deepbinner:main']},
      include_package_data=True,
      zip_safe=False,
      install_requires=['tensorflow', 'keras', 'h5py', 'numpy'],
      cmdclass={'install': DeepbinnerInstall})
