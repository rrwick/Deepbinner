

import os
import h5py
import sys


def get_read_id_and_signal(fast5_file):
    try:
        with h5py.File(fast5_file, 'r') as hdf5_file:
            read_group = list(hdf5_file['Raw/Reads/'].values())[0]
            read_id = read_group.attrs['read_id'].decode()
            signal = read_group['Signal'][:]
        return read_id, signal
    except OSError:
        return None, None


def find_all_fast5s(directory, verbose=False):
    if verbose:
        print('', file=sys.stderr)
        print('Looking for fast5 files in {}... '.format(directory), file=sys.stderr, end='',
              flush=True)
    fast5s = []
    for dir_name, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.fast5'):
                fast5s.append(os.path.join(dir_name, filename))
    if verbose:
        print(' done', file=sys.stderr)
        print('{} fast5s found'.format(len(fast5s)), file=sys.stderr)
    return fast5s
