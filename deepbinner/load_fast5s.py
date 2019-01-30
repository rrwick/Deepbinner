"""
Copyright 2018 Ryan Wick (rrwick@gmail.com)
https://github.com/rrwick/Deepbinner/

This file is part of Deepbinner. Deepbinner is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version. Deepbinner is distributed
in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along with Deepbinner.
If not, see <http://www.gnu.org/licenses/>.
"""

# Silence some warnings so the output isn't cluttered.
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', FutureWarning)

import os
import h5py
import sys


def get_read_id_and_signal(fast5_file):
    try:
        with h5py.File(fast5_file, 'r') as hdf5_file:

            # The older fast5 format (exclusively one read per file)
            if 'Raw' in hdf5_file.keys():
                read_group = list(hdf5_file['Raw/Reads/'].values())[0]

            # Newer fast5s can have multiple reads per file. Deepbinner doesn't really support this
            # yet, but it can handle it if there is just one read per file.
            else:
                reads = [x for x in hdf5_file.keys() if x.startswith('read_')]
                if len(reads) > 1:
                    sys.exit('Error: Deepbinner does not (yet) support multi-read fast5 files')
                if len(reads) == 0:
                    # TODO: print a warning here
                    return None, None
                read_name = reads[0]
                read_group = hdf5_file[read_name + '/Raw/']

            read_id = read_group.attrs['read_id'].decode()
            signal = read_group['Signal'][:]
        return read_id, signal
    except OSError:
        return None, None


def find_all_fast5s(directory, verbose=False):
    if verbose:
        print('Looking for fast5 files in {}... '.format(directory), file=sys.stderr, end='',
              flush=True)
    fast5s = []
    for dir_name, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.fast5'):
                fast5s.append(os.path.join(dir_name, filename))
    if verbose:
        noun = 'fast5' if len(fast5s) == 1 else 'fast5s'
        print('{} {} found'.format(len(fast5s), noun), file=sys.stderr)
    return fast5s
