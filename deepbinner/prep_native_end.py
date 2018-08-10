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

import sys

from .trim_signal import normalise
from .dtw_semi_global import semi_global_dtw_with_rescaling
from .prep_functions import align_read_to_reference, align_adapter_to_read_seq, trim_signal, \
    get_best_barcode, align_barcode_to_read_dtw, get_training_sample_around_signal, \
    get_training_sample_from_middle_of_signal, get_training_sample_after_signal
from . import sequences
from . import signals


ADAPTER_BARCODE_ACCEPTABLE_GAP = 4
ADAPTER_REFERENCE_ACCEPTABLE_GAP = 4
BARCODE_REFERENCE_ACCEPTABLE_GAP = 10

BARCODED_SAMPLES_PER_BARCODED_READ = 2
MIDDLE_SAMPLES_PER_BARCODED_READ = 1
NON_BARCODED_SAMPLES_PER_NON_BARCODED_READ = 2
NON_BARCODED_SAMPLE_FROM_AFTER_END_BARCODE = True


def prep_native_read_end(signal, basecalled_seq, mappy_aligner, signal_size):
    print('  sequence-based alignment', file=sys.stderr)
    print('    basecalled length: {}'.format(len(basecalled_seq)), file=sys.stderr)

    ref_start, ref_end = align_read_to_reference(basecalled_seq, mappy_aligner)
    if ref_start is None:
        return

    basecalled_end = basecalled_seq[-250:]
    offset = len(basecalled_seq) - len(basecalled_end)
    adapter_seq_start, adapter_seq_end = \
        align_adapter_to_read_seq(basecalled_end, sequences.native_end_kit_adapter, offset)
    if adapter_seq_start is None:
        return

    barcode_name, barcode_start, barcode_end = \
        get_best_barcode(basecalled_end, sequences.native_end_barcodes, offset)

    if barcode_name == 'too close':
        return

    elif barcode_name == 'none':
        if does_ref_precede_adapter(ref_end, adapter_seq_start):
            contains_barcode = False
        else:
            return

    else:  # barcode_name is 01, 02, 03, etc.
        contains_barcode = True

    print('  signal-based DTW alignment', file=sys.stderr)

    signal = trim_signal(signal)
    normalised_signal = normalise(signal)

    adapter_signal_start, adapter_signal_end = align_adapter_to_read_end_dtw(normalised_signal)
    if adapter_signal_start is None:
        return

    if contains_barcode:
        if basecalled_elements_oddly_spaced(ref_end, barcode_start, barcode_end, adapter_seq_start):
            return

        barcode_search_signal_start = adapter_signal_start - 1000
        barcode_search_signal = normalised_signal[barcode_search_signal_start:]
        barcode_signal_start, barcode_signal_end = \
            align_barcode_to_read_dtw(barcode_search_signal, barcode_search_signal_start,
                                      barcode_name, signals.native_end_barcodes)
        if barcode_signal_start is None:
            return

        if signal_elements_oddly_spaced(barcode_signal_start, barcode_signal_end,
                                        adapter_signal_start, signal_size):
            return

        make_barcoded_training_samples(barcode_name, adapter_seq_start, adapter_seq_end,
                                       barcode_start, barcode_end, ref_start, ref_end,
                                       adapter_signal_start, adapter_signal_end,
                                       barcode_signal_start, barcode_signal_end, signal,
                                       signal_size)
    else:
        make_non_barcoded_training_samples(adapter_seq_start, adapter_seq_end, ref_start, ref_end,
                                           adapter_signal_start, adapter_signal_end, signal,
                                           signal_size)


def align_adapter_to_read_end_dtw(signal):
    for i in range(0, 15000, 500):
        range_end = len(signal) - i
        range_start = range_end - 1500
        if range_start < 0:
            continue
        adapter_search_signal = signal[range_start:range_end]
        if len(adapter_search_signal) > 0:
            adapter_distance, adapter_signal_start, adapter_signal_end, _ = \
                semi_global_dtw_with_rescaling(adapter_search_signal,
                                               signals.native_end_kit_adapter)
            adapter_signal_start += range_start
            adapter_signal_end += range_start
            if adapter_distance <= 50.0:
                print('    adapter DTW: {}-{} '
                      '({:.2f})'.format(adapter_signal_start, adapter_signal_end, adapter_distance),
                      file=sys.stderr)
                return adapter_signal_start, adapter_signal_end
    else:
        print('  verdict: skipping due to high adapter DTW distance', file=sys.stderr)
        return None, None


def does_ref_precede_adapter(ref_end, adapter_seq_start):
    if abs(ref_end - adapter_seq_start) <= ADAPTER_REFERENCE_ACCEPTABLE_GAP:
        return True
    else:
        print('  verdict: skipping due to odd reference-adapter arrangement', file=sys.stderr)
        return False


def basecalled_elements_oddly_spaced(ref_end, barcode_start, barcode_end, adapter_seq_start):
    if abs(ref_end - barcode_start) > BARCODE_REFERENCE_ACCEPTABLE_GAP:
        print('  verdict: skipping due to odd reference-barcode arrangement', file=sys.stderr)
        return True
    if abs(barcode_end - adapter_seq_start) > ADAPTER_BARCODE_ACCEPTABLE_GAP:
        print('  verdict: skipping due to odd barcode-adapter arrangement', file=sys.stderr)
        return True
    return False


def signal_elements_oddly_spaced(barcode_signal_start, barcode_signal_end, adapter_signal_start,
                                 signal_size):
    if barcode_signal_end - barcode_signal_start >= signal_size:
        print('  verdict: skipping due to too-long barcode signal', file=sys.stderr)
        return True
    barcode_adapter_gap = adapter_signal_start - barcode_signal_end
    print('    barcode-adapter signal gap: {}'.format(barcode_adapter_gap), file=sys.stderr)
    print('    barcode signal size: {}'.format(barcode_signal_end - barcode_signal_start),
          file=sys.stderr)
    # TODO: tune these based on the empirical distribution (see what looks normal)
    if barcode_adapter_gap < -150 or barcode_adapter_gap > 100:
        print('  verdict: skipping due to odd adapter-barcode arrangement', file=sys.stderr)
        return True
    else:
        return False


def make_non_barcoded_training_samples(adapter_seq_start, adapter_seq_end, ref_start, ref_end,
                                       adapter_signal_start, adapter_signal_end, signal,
                                       signal_size):
    print('  verdict: good no-barcode training read', file=sys.stderr)
    print('    base coords: ref: {}-{},'
          ' adapter: {}-{}'.format(ref_start, ref_end, adapter_seq_start, adapter_seq_end),
          file=sys.stderr)
    print('    signal coords: adapter: {}-{}'.format(adapter_signal_start, adapter_signal_end),
          file=sys.stderr)
    print('  making training samples', file=sys.stderr)

    for _ in range(NON_BARCODED_SAMPLES_PER_NON_BARCODED_READ):
        training_sample = get_training_sample_around_signal(signal, adapter_signal_start - 10,
                                                            adapter_signal_start + 10, signal_size,
                                                            None)
        if training_sample is not None:
            print('0\t', end='')
            print(','.join(str(s) for s in training_sample))


def make_barcoded_training_samples(barcode_name, adapter_seq_start, adapter_seq_end, barcode_start,
                                   barcode_end, ref_start, ref_end, adapter_signal_start,
                                   adapter_signal_end, barcode_signal_start, barcode_signal_end,
                                   signal, signal_size):
    print('  verdict: good training read for barcode {}'.format(barcode_name), file=sys.stderr)
    print('    base coords: ref: {}-{}, barcode{}: {}-{}, '
          'adapter: {}-{}'.format(ref_start, ref_end, barcode_name, barcode_start, barcode_end,
                                  adapter_seq_start, adapter_seq_end), file=sys.stderr)
    print('    signal coords: barcode: {}-{}, '
          'adapter: {}-{}'.format(barcode_signal_start, barcode_signal_end,
                                  adapter_signal_start, adapter_signal_end), file=sys.stderr)
    print('  making training samples', file=sys.stderr)

    for _ in range(BARCODED_SAMPLES_PER_BARCODED_READ):
        training_sample = \
            get_training_sample_around_signal(signal, barcode_signal_start, barcode_signal_end,
                                              signal_size, barcode_name)
        if training_sample is not None:
            print('{}\t'.format(barcode_name), end='')
            print(','.join(str(s) for s in training_sample))

    if NON_BARCODED_SAMPLE_FROM_AFTER_END_BARCODE:
        training_sample = get_training_sample_after_signal(signal, adapter_signal_end + 50,
                                                           signal_size)
        if training_sample is not None:
            print('0\t', end='')
            print(','.join(str(s) for s in training_sample))

    for _ in range(MIDDLE_SAMPLES_PER_BARCODED_READ):
        training_sample = \
            get_training_sample_from_middle_of_signal(signal, signal_size)
        if training_sample is not None:
            print('0\t', end='')
            print(','.join(str(s) for s in training_sample))
