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
import sys
import pathlib
import h5py
import numpy as np
from keras.models import load_model
from .load_fast5s import find_all_fast5s, get_read_id_and_signal
from .trim_signal import normalise


def classify(args):
    start_model, start_input_size, end_model, end_input_size, output_size, model_count = \
        load_and_check_models(args.start_model, args.end_model, args.scan_size)

    input_type = determine_input_type(args.input)
    if input_type == 'training_data' and model_count == 2:
        sys.exit('Error: training data can only be classified using a single model')
    print('', file=sys.stderr)

    if input_type == 'directory':
        classify_fast5_files(find_all_fast5s(args.input, verbose=True),
                             start_model, start_input_size, end_model, end_input_size,
                             output_size, args)
    elif input_type == 'single_fast5':
        classify_fast5_files([args.input],
                             start_model, start_input_size, end_model, end_input_size,
                             output_size, args)
    elif input_type == 'training_data':
        classify_training_data(args.input, start_model, start_input_size, end_model,
                               end_input_size, output_size, args)
    else:
        assert False


def load_and_check_models(start_model_filename, end_model_filename, scan_size):
    using_read_starts = start_model_filename is not None
    using_read_ends = end_model_filename is not None
    model_count = (1 if using_read_starts else 0) + (1 if using_read_ends else 0)

    start_model, start_input_size, start_output_size = None, None, None
    if using_read_starts:
        start_model, start_input_size, start_output_size = load_trained_model(start_model_filename)
        check_input_size(start_input_size, scan_size)

    end_model, end_input_size, end_output_size = None, None, None
    if using_read_ends:
        end_model, end_input_size, end_output_size = load_trained_model(end_model_filename)
        check_input_size(end_input_size, scan_size)

    if model_count == 2:
        if start_output_size != end_output_size:
            sys.exit('Error: two models have different number of barcode classes')
        output_size = start_output_size
    elif using_read_starts:  # start only
        output_size = start_output_size
    else:  # end only
        output_size = end_output_size
    return start_model, start_input_size, end_model, end_input_size, output_size, model_count


def load_trained_model(model_file):
    if not pathlib.Path(model_file).is_file():
        sys.exit('Error: {} does not exist'.format(model_file))
    print('Loading {}... '.format(model_file), file=sys.stderr, end='', flush=True)
    model = load_model(model_file)
    print('done', file=sys.stderr)
    try:
        assert len(model.inputs) == 1
        input_shape = model.inputs[0].shape
        output_shape = model.outputs[0].shape
        input_size = int(input_shape[1])
        output_size = int(output_shape[1])
        assert input_size > 10
        assert input_shape[2] == 1
    except (AssertionError, IndexError):
        sys.exit('Error: model input has incorrect shape - are you sure that {} is a valid '
                 'model file?'.format(model_file))
    return model, input_size, output_size


def classify_fast5_files(fast5_files, start_model, start_input_size, end_model, end_input_size,
                         output_size, args, summary_table=True):
    using_read_starts = start_model is not None
    using_read_ends = end_model is not None

    print_classification_progress(0, len(fast5_files), 'fast5s')
    print_output_header(args.verbose, using_read_starts, using_read_ends)

    classifications, read_id_to_fast5_file = {}, {}
    for fast5_batch in chunker(fast5_files, args.batch_size):
        read_ids, signals = [],  []

        for i, fast5_file in enumerate(fast5_batch):
            read_id, signal = get_read_id_and_signal(fast5_file)
            read_id_to_fast5_file[read_id] = fast5_file
            read_ids.append(read_id)
            signals.append(signal)

        start_calls, start_probs = None, None
        if using_read_starts:
            start_calls, start_probs = call_batch(start_input_size, output_size, read_ids, signals,
                                                  start_model, args, 'start')
        end_calls, end_probs = None, None
        if using_read_ends:
            end_calls, end_probs = call_batch(end_input_size, output_size, read_ids, signals,
                                              end_model, args, 'end')

        for i, read_id in enumerate(read_ids):
            if using_read_starts and using_read_ends:
                final_barcode_call = combine_calls(start_calls[i], end_calls[i], args.require_both)
            elif using_read_starts:  # starts only
                final_barcode_call = start_calls[i]
            else:  # ends only
                final_barcode_call = end_calls[i]
            output = [read_id, final_barcode_call]
            classifications[read_id] = final_barcode_call

            if args.verbose:
                output += ['%.2f' % x for x in start_probs[i]]
                if using_read_ends:
                    output.append(start_calls[i])
                    output += ['%.2f' % x for x in end_probs[i]]
                    output.append(end_calls[i])
            print('\t'.join(output))

        print_classification_progress(len(classifications), len(fast5_files), 'fast5s')

    if summary_table:
        print('', file=sys.stderr)
        print_summary_table(classifications)
    return classifications, read_id_to_fast5_file


def classify_training_data(input_file, start_model, start_input_size, end_model, end_input_size,
                           output_size, args):
    using_read_starts = start_model is not None
    using_read_ends = end_model is not None

    num_lines = sum(1 for _ in open(input_file))
    print_classification_progress(0, num_lines, 'training data')

    print_output_header(args.verbose, using_read_starts, using_read_ends)

    assert not(using_read_starts and using_read_ends)
    if using_read_starts:
        model = start_model
        input_size = start_input_size
    else:
        model = end_model
        input_size = end_input_size

    classifications = {}
    with open(input_file, 'rt') as training_data:
        line_num = 0
        finished = False
        while True:

            # Read in a batch of lines.
            read_ids, signals = [], []
            while True:
                try:
                    line = next(training_data).rstrip()
                except StopIteration:
                    finished = True
                    break
                line_num += 1
                barcode, signal = line.split('\t')
                read_id = 'line_{}_barcode_{}'.format(line_num, barcode)
                read_ids.append(read_id)
                signals.append(np.array([int(x) for x in signal.split(',')]))
                if len(read_ids) == args.batch_size:
                    break

            start_calls, start_probs = call_batch(input_size, output_size, read_ids, signals,
                                                  model, args, 'start')

            for i, read_id in enumerate(read_ids):
                final_barcode_call = start_calls[i]
                output = [read_id, final_barcode_call]
                classifications[read_id] = final_barcode_call
                if args.verbose:
                    output += ['%.2f' % x for x in start_probs[i]]
                print('\t'.join(output))

            print_classification_progress(len(classifications), num_lines, 'training data')
            if finished:
                break

    print('', file=sys.stderr)
    print_summary_table(classifications)


def determine_input_type(input_file_or_dir):
    path = pathlib.Path(input_file_or_dir)
    if path.is_dir():
        return 'directory'
    if not path.is_file():
        sys.exit('Error: {} is neither a file nor a directory'.format(input_file_or_dir))
    try:
        f = h5py.File(input_file_or_dir, 'r')
        f.close()
        return 'single_fast5'
    except OSError:
        pass
    with open(input_file_or_dir) as f:
        first_line = f.readline()
    try:
        parts = first_line.split('\t')
        _ = int(parts[0])
        signals = [int(x) for x in parts[1].split(',')]
        assert len(signals) > 10
        return 'training_data'
    except (AssertionError, ValueError, IndexError):
        sys.exit('Error: could not determine input type')


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def print_output_header(verbose, using_read_starts, using_read_ends):
    if not verbose:
        print('\t'.join(['read_ID', 'barcode_call']))

    elif using_read_starts and using_read_ends:
        print('\t'.join(['read_ID', 'barcode_call',
                         'start_none', 'start_1', 'start_2', 'start_3', 'start_4', 'start_5',
                         'start_6', 'start_7', 'start_8', 'start_9', 'start_10', 'start_11',
                         'start_12', 'start_barcode_call',
                         'end_none', 'end_1', 'end_2', 'end_3', 'end_4', 'end_5', 'end_6', 'end_7',
                         'end_8', 'end_9', 'end_10', 'end_11', 'end_12', 'end_barcode_call']))
    else:  # just doing starts or ends
        print('\t'.join(['read_ID', 'barcode_call',
                         'none', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']))


def get_barcode_call_from_probabilities(probabilities, score_diff_threshold):
    probabilities = list(enumerate(probabilities))  # make into tuples (barcode, prob)
    probabilities = sorted(probabilities, key=lambda x: x[1], reverse=True)
    best, second_best = probabilities[0], probabilities[1]
    if best[0] == 0:
        return 'none'
    score_diff = best[1] - second_best[1]
    if score_diff >= score_diff_threshold:
        return str(best[0])
    else:
        return 'none'


def combine_calls(start_call, end_call, require_both):
    if start_call == end_call:
        return start_call
    if require_both:
        return 'none'
    if start_call == 'none':
        return end_call
    if end_call == 'none':
        return start_call
    return 'none'


def call_batch(input_size, output_size, read_ids, signals, model, args, side):
    probabilities = []
    for _ in read_ids:
        probabilities.append([0.0] * output_size)

    step_size = input_size // 2
    steps = int(args.scan_size / step_size)

    # TO DO: check to make sure this will work earlier in the code and quit with a nice error
    # message if not so.
    assert steps * step_size == args.scan_size

    for s in range(steps):
        sig_start = s * step_size
        sig_end = sig_start + input_size

        input_signals = np.empty([len(read_ids), input_size], dtype=float)
        for i, signal in enumerate(signals):
            if side == 'start':
                input_signal = signal[sig_start:sig_end]
            else:
                assert side == 'end'
                a = max(len(signal) - sig_end, 0)
                b = max(len(signal) - sig_start, 0)
                input_signal = signal[a:b]

            input_signal = normalise(input_signal)
            if len(input_signal) < input_size:
                pad_size = input_size - len(input_signal)
                if side == 'start':
                    input_signal = np.pad(input_signal, (0, pad_size), 'constant')
                else:
                    input_signal = np.pad(input_signal, (pad_size, 0), 'constant')
            input_signals[i] = input_signal

        input_signals = np.expand_dims(input_signals, axis=2)
        labels = model.predict(input_signals, batch_size=args.batch_size)

        # For each read, we keep the new set of probabilities if they have a stronger match for
        # any of the barcodes (excluding the none class).
        for i, read_id in enumerate(read_ids):
            if max(labels[i][1:]) > max(probabilities[i][1:]):
                probabilities[i] = labels[i]

    barcode_calls = []
    for i, read_id in enumerate(read_ids):
        barcode_calls.append(get_barcode_call_from_probabilities(probabilities[i], args.score_diff))

    return barcode_calls, probabilities


def check_input_size(input_size, scan_size):
    step_size = input_size // 2
    if step_size * 2 != input_size:  # if input_size is not an even number:
        sys.exit('Error: the model input size must be even (currently {})'.format(input_size))

    steps = int(scan_size / step_size)
    if steps * step_size != scan_size:
        acceptable_scan_sizes = [str(step_size * i) for i in range(2, 8)]
        acceptable_scan_sizes.append('etc')
        sys.exit('Error: --scan_size must be a multiple of half the model input size\n'
                 'acceptable values for --scan_size are '
                 '{}'.format(', '.join(acceptable_scan_sizes)))


def print_classification_progress(completed, total, label):
    percent = 100.0 * completed / total
    print('\rClassifying {}: {} / {} ({:.1f}%)'.format(label, completed, total, percent),
          file=sys.stderr, end='', flush=True)


def print_summary_table(classifications):
    counts = collections.defaultdict(int)
    for barcode in classifications.values():
        counts[barcode] += 1
    print('', file=sys.stderr)
    print('Barcode     Count', file=sys.stderr)

    number_barcodes, string_barcodes = [], []
    for barcode in counts.keys():
        try:
            number_barcodes.append(int(barcode))
        except ValueError:
            string_barcodes.append(barcode)
    sorted_barcodes = sorted(number_barcodes) + sorted(string_barcodes)

    for barcode in sorted_barcodes:
        print('{:>7} {:>9}'.format(barcode, counts[str(barcode)]), file=sys.stderr)
    print('', file=sys.stderr)
