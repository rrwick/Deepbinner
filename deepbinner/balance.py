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
import sys


def balance_training_samples(args):
    signal_size = get_signal_size(args.training_data)
    counts = count_samples_all_runs(args.training_data)
    barcodes = get_barcodes(counts, args.barcodes)
    smallest_count = get_smallest_count(barcodes, counts)
    used_samples_per_run = get_used_samples_per_run(barcodes, counts, smallest_count)
    select_samples(args.training_data, used_samples_per_run)
    add_random_signals(args.random_signal, smallest_count, signal_size)
    print('', file=sys.stderr)


def get_signal_size(training_data_filenames):
    signal_sizes = set()
    for training_data_filename in training_data_filenames:
        signal_sizes.add(get_signal_size_one_run(training_data_filename))
    signal_sizes = sorted(signal_sizes)
    if len(signal_sizes) == 1:
        signal_size = signal_sizes[0]
        print('\nSignal size detected as {}'.format(signal_size), file=sys.stderr)
        return signal_size
    else:
        sys.exit('Error: multiple signal sizes detected: {}'.format(signal_sizes))


def get_signal_size_one_run(training_data_filename):
    with open(training_data_filename, 'rt') as training_data:
        for line in training_data:
            parts = line.strip().split('\t')
            signal = parts[1].split(',')
            return len(signal)


def count_samples_all_runs(training_data_filenames):
    print('\nCounting samples in input files:', file=sys.stderr)
    counts_per_run = {}
    for training_data_filename in training_data_filenames:
        counts_per_run[training_data_filename] = count_samples_one_run(training_data_filename)
    return counts_per_run


def count_samples_one_run(training_data_filename):
    counts = collections.defaultdict(int)
    with open(training_data_filename, 'rt') as training_data:
        for line in training_data:
            barcode = int(line.split('\t', maxsplit=1)[0])
            counts[barcode] += 1

    count_str = ', '.join(str(b) + '=' + str(counts[b]) for b in sorted(counts.keys()))
    print('  {}: {}'.format(training_data_filename, count_str), file=sys.stderr)
    return counts


def get_barcodes(all_run_counts, user_supplied_barcodes):
    if user_supplied_barcodes is None:
        barcodes = set()
        for counts in all_run_counts.values():
            for barcode in counts.keys():
                barcodes.add(barcode)
        barcodes = sorted(barcodes)
        barcode_str = ', '.join(str(x) for x in barcodes)
        print('\nIncluding all barcodes in output: {}'.format(barcode_str), file=sys.stderr)
        return barcodes
    else:
        barcodes = [int(x) for x in user_supplied_barcodes]
        if 0 not in barcodes:
            barcodes.append(0)
        barcodes = sorted(barcodes)
        barcode_str = ', '.join(str(x) for x in barcodes)
        print('\nIncluding user-specified barcodes in output: {}'.format(barcode_str),
              file=sys.stderr)
        return barcodes


def get_smallest_count(barcodes, counts):
    barcode_counts = []
    for barcode in barcodes:
        barcode_counts.append(sum(counts[run][barcode] for run in counts))
    smallest_count = min(barcode_counts)
    print('\nSmallest count = {}'.format(smallest_count), file=sys.stderr)
    print('  all barcodes will be limited to this many samples', file=sys.stderr)
    return smallest_count


def get_used_samples_per_run(barcodes, counts, smallest_count):
    print('\nDetermining how many samples to take from each file:', file=sys.stderr)
    used_samples_per_run = {}
    for barcode in barcodes:
        used_samples_per_run[barcode] = get_used_samples_per_barcode(barcode, counts,
                                                                     smallest_count)
    return used_samples_per_run


def get_used_samples_per_barcode(barcode, counts, smallest_count):
    barcode_name = 'no barcode' if barcode == 0 else 'barcode {:02d}'.format(barcode)
    print('  {}: '.format(barcode_name), file=sys.stderr)
    used_samples = collections.defaultdict(int)

    total = 0
    while total < smallest_count:
        for run in counts:
            if used_samples[run] < counts[run][barcode]:
                used_samples[run] += 1
                total += 1
                if total == smallest_count:
                    break
    for run, count in used_samples.items():
        print('    {}: {}'.format(run, count), file=sys.stderr)
    return used_samples


def select_samples(training_data_filenames, used_samples_per_run):
    print('\nSelecting samples and printing to stdout:', file=sys.stderr)
    for training_data_filename in training_data_filenames:
        select_samples_one_run(training_data_filename, used_samples_per_run)


def select_samples_one_run(training_data_filename, used_samples_per_run):
    print('  {}'.format(training_data_filename), file=sys.stderr)
    samples_per_barcode = collections.defaultdict(list)
    with open(training_data_filename, 'rt') as training_data:
        for line in training_data:
            barcode = int(line.split('\t', maxsplit=1)[0])
            samples_per_barcode[barcode].append(line)
    for barcode in samples_per_barcode:
        random.shuffle(samples_per_barcode[barcode])
    for barcode in sorted(used_samples_per_run):
        barcode_name = 'no barcode' if barcode == 0 else 'barcode {:02d}'.format(barcode)
        number_of_samples = used_samples_per_run[barcode][training_data_filename]
        print('    {}: {}'.format(barcode_name, number_of_samples), file=sys.stderr)
        for sample in samples_per_barcode[barcode][:number_of_samples]:
            print(sample, end='')


def add_random_signals(random_amount, smallest_count, signal_size):
    random_count = int(round(random_amount * smallest_count))
    print('\nAdding {} random signals as no-barcode training samples'.format(random_count),
          file=sys.stderr)
    for _ in range(random_count):
        print('0\t', end='')
        print(get_random_signal(signal_size))


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
