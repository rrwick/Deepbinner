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

import collections
import random
import noise
from .trim_signal import clean_signal


def balance_training_samples(args):
    bin_counts, bin_lines = load_data_by_bin(args.training_data, args.barcodes)
    smallest_count = min(bin_counts.values())

    bins = sorted(bin_counts.keys(), key=lambda x: int(x))
    total_count_excluding_nones = len(bins) * smallest_count
    total_count = int(round(total_count_excluding_nones / (1.0 - args.none_bin_rate)))
    none_count = total_count - total_count_excluding_nones

    print()
    print('Producing ' + str(smallest_count) + ' samples for each bin')
    print('  and ' + str(none_count) + ' samples with no barcode')
    print('  for a total of ' + str(total_count) + ' samples')

    # Half of the 'none' samples will be signal from the middle of reads. The remainder will be
    # randomly generated (in a variety of ways) signal.
    middle_signal_none_count = int(0.5 * none_count)
    random_none_count = none_count - middle_signal_none_count

    start_filename = args.out_prefix + '_read_starts'
    end_filename = args.out_prefix + '_read_ends'

    with open(start_filename, 'w') as start_read_file, open(end_filename, 'w') as end_read_file:
        middle_signals = []
        for barcode_bin in bins:
            for line in random.sample(bin_lines[barcode_bin], k=smallest_count):
                parts = line.split('\t')
                assert barcode_bin == parts[1]
                start_signal, middle_signal, end_signal = parts[2], parts[3], parts[4]

                start_signal, end_signal, good_start, good_end = \
                    clean_signal(start_signal, end_signal, args.signal_size, args.plot)

                if good_start:
                    start_read_file.write(barcode_bin)
                    start_read_file.write('\t')
                    start_read_file.write(start_signal)
                    start_read_file.write('\n')

                if good_end:
                    end_read_file.write(barcode_bin)
                    end_read_file.write('\t')
                    end_read_file.write(end_signal)
                    end_read_file.write('\n')

                middle_signals.append(middle_signal)

        for middle_signal in random.sample(middle_signals, k=middle_signal_none_count):
            start_read_file.write('0\t')
            start_read_file.write(middle_signal)
            start_read_file.write('\n')

        for middle_signal in random.sample(middle_signals, k=middle_signal_none_count):
            end_read_file.write('0\t')
            end_read_file.write(middle_signal)
            end_read_file.write('\n')

        for _ in range(random_none_count):
            start_read_file.write('0\t')
            start_read_file.write(get_random_signal(args.signal_size))
            start_read_file.write('\n')
            end_read_file.write('0\t')
            end_read_file.write(get_random_signal(args.signal_size))
            end_read_file.write('\n')

    print()


def load_data_by_bin(raw_training_data_filename, barcodes):
    print()
    print('Loading raw training data... ', end='', flush=True)
    bin_counts = collections.defaultdict(int)
    bin_lines = collections.defaultdict(list)
    with open(raw_training_data_filename, 'rt') as raw_training_data:
        for line in raw_training_data:
            if line.startswith('Read_ID'):
                continue
            barcode_bin = line.split('\t')[1]
            if barcodes is None or barcode_bin in barcodes:
                bin_counts[barcode_bin] += 1
                bin_lines[barcode_bin].append(line.strip())
    print('done')

    print()
    print('Bin        Samples')
    print('------------------')
    bin_nums = sorted(int(x) for x in bin_counts.keys())
    for bin_num in bin_nums:
        print('%2d' % bin_num, end='')
        print('%16s' % bin_counts[str(bin_num)])
    return bin_counts, bin_lines


def get_random_signal(signal_size):
    random_type = random.choice(['flat', 'gaussian', 'multi_gaussian', 'perlin'])
    signal = []

    if random_type == 'flat':
        signal = [random.randint(0, 1000)] * signal_size

    elif random_type == 'gaussian':
        mean = random.uniform(300, 600)
        stdev = random.uniform(10, 500)
        signal = [int(random.gauss(mean, stdev)) for _ in range(signal_size)]

    elif random_type == 'multi_gaussian':
        while len(signal) < signal_size:
            length = random.randint(100, 500)
            mean = random.uniform(300, 600)
            stdev = random.uniform(10, 500)
            signal += [int(random.gauss(mean, stdev)) for _ in range(length)]

    elif random_type == 'perlin':
        octaves = random.randint(1, 4)
        step = random.uniform(0.001, 0.04)
        start = random.uniform(0.0, 1.0)
        factor = random.uniform(10, 300)
        mean = random.uniform(300, 600)
        signal = [int(mean + factor * noise.pnoise1(start + (i * step), octaves))
                  for i in range(signal_size)]

    signal = signal[:signal_size]
    return ','.join(str(x) for x in signal)
