

import sys
import os
import h5py

from .trim_signal import too_much_open_pore


def get_signal_from_fast5s(fast5_dir, signal_size, max_start_end_margin, min_signal_length):
    fast5_files = find_all_fast5s(fast5_dir)
    print('Loading signal from ' + str(len(fast5_files)) + ' fast5 files',
          end='', file=sys.stderr)
    signals = {}
    i = 0

    short_count = 0
    bad_signal_count = 0

    for fast5_file in fast5_files:
        i += 1
        if i % 100 == 0:
            print('.', end='', file=sys.stderr, flush=True)

        read_id, signal = get_read_id_and_signal(fast5_file)

        if len(signal) < min_signal_length:
            short_count += 1
            continue

        bad_signal = False

        # The middle signal is simply the centre-most signal in the read.
        middle_pos = len(signal) // 2
        middle_1 = middle_pos - (signal_size // 2)
        middle_2 = middle_pos + (signal_size // 2)
        middle_signal = signal[middle_1:middle_2]

        start_margin = signal_size * 2
        while too_much_open_pore(signal[:start_margin]):
            start_margin += signal_size
            if start_margin > max_start_end_margin:
                bad_signal = True
                break

        end_margin = signal_size * 2
        while too_much_open_pore(signal[-end_margin:]):
            end_margin += signal_size
            if end_margin > max_start_end_margin:
                bad_signal = True
                break

        if bad_signal:
            bad_signal_count += 1
            continue

        start_signal = signal[:start_margin]
        end_signal = signal[-end_margin:]

        signals[read_id] = (start_signal, middle_signal, end_signal)

    print(' done', file=sys.stderr)
    print('skipped ' + str(short_count) + ' reads for being too short\n', file=sys.stderr)
    print('skipped ' + str(bad_signal_count) + ' reads for bad signal stdev\n', file=sys.stderr)
    return signals


def get_read_id_and_signal(fast5_file):
    with h5py.File(fast5_file, 'r') as hdf5_file:
        read_group = list(hdf5_file['Raw/Reads/'].values())[0]
        read_id = read_group.attrs['read_id'].decode()
        signal = read_group['Signal'][:]
    return read_id, signal


def find_all_fast5s(directory):
    fast5s = []
    for dir_name, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.fast5'):
                fast5s.append(os.path.join(dir_name, filename))
    return fast5s
