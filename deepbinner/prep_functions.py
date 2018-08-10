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

import edlib
import random
import re
import sys

from .trim_signal import find_signal_start_pos, CannotTrim
from .dtw_semi_global import semi_global_dtw_with_rescaling


MIN_ADAPTER_IDENTITY = 70.0
MIN_BARCODE_IDENTITY = 70.0
MIN_BEST_SECOND_BEST_DIFF = 7.5
MIN_REFERENCE_IDENTITY = 70.0
MIN_READ_COVERAGE = 70.0
MIN_BASECALLED_LENGTH = 500


def align_read_to_reference(basecalled_seq, mappy_aligner):
    if len(basecalled_seq) < MIN_BASECALLED_LENGTH:
        print('  verdict: skipping due to short basecalled length', file=sys.stderr)
        return None, None
    ref_id, read_cov, ref_start, ref_end = minimap_align(basecalled_seq, mappy_aligner)
    if ref_id == 0.0:
        print('  verdict: skipping due to no alignment to reference', file=sys.stderr)
        return None, None
    elif ref_id < MIN_REFERENCE_IDENTITY:
        print('  verdict: skipping due to low reference alignment identity', file=sys.stderr)
        return None, None
    elif read_cov < MIN_READ_COVERAGE:
        print('  verdict: skipping due to short reference alignment', file=sys.stderr)
        return None, None
    else:
        print('    reference seq: {}-{} ({:.1f}%)'.format(ref_start, ref_end, ref_id),
              file=sys.stderr)
        return ref_start, ref_end


def minimap_align(query_seq, aligner):
    best_hit, best_mlen = None, 0
    for hit in aligner.map(query_seq):
        if hit.mlen > best_mlen:
            best_hit, best_mlen = hit, hit.mlen
    if best_hit is None:
        return 0.0, 0.0, 0, 0
    identity = 100.0 * best_hit.mlen / best_hit.blen
    read_start, read_end = best_hit.q_st, best_hit.q_en
    read_cov = 100.0 * (read_end - read_start) / len(query_seq)
    return identity, read_cov, read_start, read_end


def align_adapter_to_read_seq(basecalled_start, adapter, offset=0):
    adapter_identity, adapter_start, adapter_end = edlib_align(adapter, basecalled_start)
    adapter_start += offset
    adapter_end += offset
    print('    adapter seq: {}-{} ({:.1f}%)'.format(adapter_start, adapter_end, adapter_identity),
          file=sys.stderr)
    if adapter_identity < MIN_ADAPTER_IDENTITY:
        print('  verdict: skipping due to low adapter alignment identity', file=sys.stderr)
        return None, None
    else:
        return adapter_start, adapter_end


def trim_signal(signal):
    print('    untrimmed signal length: {}'.format(len(signal)), file=sys.stderr)
    try:
        start_trim_pos = find_signal_start_pos(signal)
    except CannotTrim:
        print('  verdict: skipping due to failed signal trimming', file=sys.stderr)
        return None
    print('    trim amount: {}'.format(start_trim_pos), file=sys.stderr)
    return signal[start_trim_pos:]


def edlib_align(query_seq, ref_seq):
    alignment = edlib.align(query_seq, ref_seq, mode='HW', task='path')
    return (identity_from_edlib_cigar(alignment['cigar']),
            alignment['locations'][0][0], alignment['locations'][0][1])


def identity_from_edlib_cigar(cigar):
    matches, alignment_length = 0, 0
    cigar_parts = re.findall(r'\d+[IDX=]', cigar)
    for c in cigar_parts:
        cigar_type = c[-1]
        cigar_size = int(c[:-1])
        alignment_length += cigar_size
        if cigar_type == '=':
            matches += cigar_size
    try:
        return 100.0 * matches / alignment_length
    except ZeroDivisionError:
        return 0.0


def get_best_barcode(read_seq, barcode_seqs, offset=0):
    best_barcode_name, best_barcode_identity = None, 0.0
    best_start, best_end = 0, 0
    all_identities = []
    for barcode_name, barcode_seq in barcode_seqs.items():
        barcode_identity, barcode_start, barcode_end = edlib_align(barcode_seq, read_seq)
        if barcode_identity > best_barcode_identity:
            best_barcode_name, best_barcode_identity = barcode_name, barcode_identity
            best_start, best_end = barcode_start, barcode_end
        all_identities.append(barcode_identity)
    all_identities = sorted(all_identities)
    best_second_best_diff = all_identities[-1] - all_identities[-2]
    best_start += offset
    best_end += offset
    if best_barcode_identity < MIN_BARCODE_IDENTITY:
        return 'none', best_start, best_end
    if best_second_best_diff < MIN_BEST_SECOND_BEST_DIFF:
        print('  verdict: skipping due to too-close-to-call barcodes', file=sys.stderr)
        return 'too close', best_start, best_end
    else:
        print('    best barcode: #{}, {}-{} ({:.2f}%)'.format(best_barcode_name, best_start,
                                                              best_end, best_barcode_identity),
              file=sys.stderr)
        return best_barcode_name, best_start, best_end


def align_barcode_to_read_dtw(barcode_search_signal, barcode_search_signal_start, barcode_name,
                              barcode_signals):
    if barcode_search_signal_start < 0:
        return None, None
    barcode_distance, barcode_signal_start, barcode_signal_end, _ = \
        semi_global_dtw_with_rescaling(barcode_search_signal, barcode_signals[barcode_name])
    barcode_signal_start += barcode_search_signal_start
    barcode_signal_end += barcode_search_signal_start
    print('    barcode{} DTW: {}-{} ({:.2f})'.format(barcode_name, barcode_signal_start,
                                                     barcode_signal_end, barcode_distance),
          file=sys.stderr)
    if barcode_distance > 50.0:
        print('  verdict: skipping due to high barcode DTW distance', file=sys.stderr)
        return None, None
    else:
        return barcode_signal_start, barcode_signal_end


def get_training_sample_around_signal(signal, include_start, include_end, signal_size,
                                      barcode_name):
    """
    This function takes in a large signal and returns a training-sized chunk which includes the
    specified range.
    """
    include_size = include_end - include_start
    min_start = max(0, include_start + include_size - signal_size)
    training_start = random.randint(min_start, include_start)
    training_end = training_start + signal_size

    if barcode_name is None:
        print('    no-barcode sample taken from trimmed signal: '
              '{}-{}'.format(training_start, training_end), file=sys.stderr)
    else:
        print('    barcode {} sample taken from trimmed signal: '
              '{}-{}'.format(barcode_name, training_start, training_end), file=sys.stderr)
    return signal[training_start:training_end]


def get_training_sample_from_middle_of_signal(signal, signal_size):
    """
    This function takes in a large signal and returns a training-sized chunk from the middle.
    """
    try:
        training_start = random.randint(25000, len(signal) - 25000 - signal_size)
    except ValueError:
        return None
    training_end = training_start + signal_size
    print('    no-barcode sample taken from middle of signal: '
          '{}-{}'.format(training_start, training_end), file=sys.stderr)
    return signal[training_start:training_end]


def get_training_sample_before_signal(signal, before_point, signal_size):
    """
    This function takes in a large signal and returns a training-sized chunk which occurs
    before the given point.
    """
    try:
        training_start = random.randint(0, before_point - signal_size)
    except ValueError:
        return None
    training_end = training_start + signal_size
    print('    no-barcode sample taken from trimmed signal start: '
          '{}-{}'.format(training_start, training_end), file=sys.stderr)
    return signal[training_start:training_end]


def get_training_sample_after_signal(signal, after_point, signal_size):
    """
    This function takes in a large signal and returns a training-sized chunk which occurs
    after the given point.
    """
    try:
        training_end = random.randint(after_point + signal_size, len(signal))
    except ValueError:
        return None
    training_start = training_end - signal_size
    print('    no-barcode sample taken from signal end: '
          '{}-{}'.format(training_start, training_end), file=sys.stderr)
    return signal[training_start:training_end]


def albacore_barcode_agrees(barcode_name, albacore_barcode):
    if albacore_barcode is None:
        return True
    if barcode_name == 'none' or albacore_barcode == 'unclassified':
        return True

    if barcode_name == albacore_barcode:
        return True
    else:
        print('  verdict: skipping because albacore barcode ({})'
              ' disagrees'.format(albacore_barcode), file=sys.stderr)
        return False
