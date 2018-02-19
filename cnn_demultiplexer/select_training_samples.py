
import sys
import collections
import random


def select_training_samples(args):
    raw_training_data_filename = sys.argv[1]
    output_prefix = sys.argv[2]

    bin_counts, bin_lines = load_data_by_bin(raw_training_data_filename)
    smallest_count = min(bin_counts.values())

    bins = sorted(bin_counts.keys(), key=lambda x: int(x))
    total_count_excluding_nones = len(bins) * smallest_count
    total_count = int(round(total_count_excluding_nones / (1.0 - args.none_bin_rate)))
    none_count = total_count - total_count_excluding_nones

    print()
    print('Producing ' + str(smallest_count) + ' samples for each bin')
    print('  and ' + str(none_count) + ' samples with no barcode')
    print('  for a total of ' + str(total_count) + ' samples')

    start_filename = output_prefix + '_read_starts'
    end_filename = output_prefix + '_read_ends'

    with open(start_filename, 'w') as start_read_file, open(end_filename, 'w') as end_read_file:
        middle_signals = []
        for barcode_bin in bins:
            for line in random.sample(bin_lines[barcode_bin], k=smallest_count):
                parts = line.split('\t')
                assert barcode_bin == parts[1]
                start_signal, middle_signal, end_signal = parts[4], parts[5], parts[6]

                start_read_file.write(barcode_bin)
                start_read_file.write('\t')
                start_read_file.write(start_signal)
                start_read_file.write('\n')

                end_read_file.write(barcode_bin)
                end_read_file.write('\t')
                end_read_file.write(end_signal)
                end_read_file.write('\n')

                middle_signals.append(middle_signal)

        for middle_signal in random.sample(middle_signals, k=none_count):
            start_read_file.write('0\t')
            start_read_file.write(middle_signal)
            start_read_file.write('\n')

        for middle_signal in random.sample(middle_signals, k=none_count):
            end_read_file.write('0\t')
            end_read_file.write(middle_signal)
            end_read_file.write('\n')
    
    print()


def load_data_by_bin(raw_training_data_filename):
    bin_counts = collections.defaultdict(int)
    bin_lines = collections.defaultdict(list)
    with open(raw_training_data_filename, 'rt') as raw_training_data:
        for line in raw_training_data:
            if line.startswith('Read_ID'):
                continue
            barcode_bin = line.split('\t')[1]
            bin_counts[barcode_bin] += 1
            bin_lines[barcode_bin].append(line.strip())
    return bin_counts, bin_lines
