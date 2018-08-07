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
import mappy as mp
import pathlib
import random
import re

from .load_fast5s import get_read_id_and_signal, find_all_fast5s
from .misc import load_fastq
from .trim_signal import normalise
from .dtw import semi_global_dtw_with_rescaling
from . import sequences
from . import signals


MIN_ADAPTER_IDENTITY = 70.0
MIN_BARCODE_IDENTITY = 70.0
MIN_BEST_SECOND_BEST_DIFF = 5.0
MIN_REFERENCE_IDENTITY = 70.0
MIN_READ_COVERAGE = 70.0
ACCEPTABLE_GAP = 4


def prep(args):
    if pathlib.Path(args.fast5_dir).is_dir():
        fast5s = find_all_fast5s(args.fast5_dir)
    else:
        fast5s = [args.fast5_dir]
    fast5s = fast5s[:500]  # TEMP

    read_seqs = load_fastq(args.fastq)

    if args.kit == 'EXP-NBD103':
        mappy_aligner = mp.Aligner(args.ref_fasta)
    else:
        mappy_aligner = None

    for fast5_file in fast5s:
        read_id, signal = get_read_id_and_signal(fast5_file)
        if read_id not in read_seqs:
            continue

        print()
        print(fast5_file)
        print('  read ID: {}'.format(read_id))

        if args.kit == 'EXP-NBD103' and args.start_end == 'start':
            prep_native_read_start(read_id, signal, read_seqs[read_id], mappy_aligner,
                                   args.signal_size)
        if args.kit == 'EXP-NBD103' and args.start_end == 'end':
            prep_native_read_end()
        elif args.kit == 'SQK-RBK004' and args.start_end == 'start':
            prep_rapid_read_start()


def prep_native_read_start(read_id, signal, basecalled_seq, mappy_aligner, signal_size):

    # First, we make sure the read aligns to the reference.
    ref_id, read_cov, ref_start, ref_end = minimap_align(basecalled_seq, mappy_aligner)
    if ref_id < MIN_REFERENCE_IDENTITY or read_cov < MIN_READ_COVERAGE:
        print('SKIPPING: poor alignment to reference')
        return

    # Next, we look for the adapter sequence at the start of the basecalled read.
    basecalled_start = basecalled_seq[:500]
    adapter_identity, adapter_start, adapter_end = edlib_align(sequences.native_start_kit_adapter,
                                                               basecalled_start)
    print('  adapter seq: {}-{} ({:.1f}%)'.format(adapter_start, adapter_end, adapter_identity))
    if adapter_identity < MIN_ADAPTER_IDENTITY:
        print('SKIPPING: adapter aligned with low identity')
        return

    # Now we look for the adapter signal in the read signal using DTW.
    signal = normalise(signal)
    for i in range(0, 20000, 500):
        adapter_search_signal = signal[i:i+1500]
        if len(adapter_search_signal) > 0:
            adapter_distance, adapter_signal_start, adapter_signal_end, _ = \
                semi_global_dtw_with_rescaling(adapter_search_signal, signals.native_start_kit_adapter)
            adapter_signal_start += i
            adapter_signal_end += i
            print('  adapter DTW: {}-{} ({:.2f})'.format(adapter_signal_start, adapter_signal_end,
                                                         adapter_distance))
            if adapter_distance <= 100.0:
                break
    else:
        print('SKIPPING: adapter aligned with high DTW distance')
        return

    # Now look for a good barcode in the read sequence.
    barcode_name, barcode_identity, barcode_start, barcode_end = \
        get_best_barcode(basecalled_start, sequences.native_start_barcodes)
    print('  best barcode: #{}, {}-{} ({:.2f}%)'.format(barcode_name, barcode_start, barcode_end,
                                                        barcode_identity))

    # If there isn't a barcode and the reference sequence follows the ligation adapter, then this
    # is a genuine no-barcode read.
    if (barcode_name == 'none' or barcode_name == 'too close') and \
            abs(adapter_end - ref_start) <= ACCEPTABLE_GAP:
        print('GOOD NO-BARCODE TRAINING READ')
        print('  base coords: adapter: {}-{} ({:.1f}%),'
              ' ref: {}-{}'.format(adapter_start, adapter_end, adapter_identity,
                                   ref_start, ref_end))
        print('  signal coords: adapter: {}-{}'.format(adapter_signal_end, adapter_signal_end))
        training_sample = get_training_sample_from_signal(signal, adapter_signal_end - 10,
                                                          adapter_signal_end + 10, signal_size)
        # TODO: print sample
        return

    # See if the arrangement of elements in the basecalled read looks too weird.
    elif abs(adapter_end - barcode_start) > ACCEPTABLE_GAP or \
            abs(barcode_end - ref_start) > ACCEPTABLE_GAP:
        print('SKIPPING: read elements oddly arranged')
        return

    # If the arrangement is good, then we check for the barcode in the signal with DTW.
    barcode_search_signal = signal[adapter_signal_end - 100:adapter_signal_end + 1000]
    dtw_barcode_name, barcode_signal_start, barcode_signal_end = \
        get_best_barcode_dtw(barcode_search_signal, signals.native_start_barcodes,
                             adapter_signal_end)

    # If the sequence-based and DTW-based barcodes disagree, then we won't use this read.
    if dtw_barcode_name != barcode_name:
        print('SKIPPING: seq barcode and DTW barcode disagree')
        return

    # See if the arrangement of elements in the read signal looks too weird.
    adapter_barcode_gap = barcode_signal_start - adapter_signal_end
    print('  adapter-barcode signal gap: {}'.format(adapter_barcode_gap))
    print('  barcode signal size: {}'.format(barcode_signal_end - barcode_signal_start))
    if adapter_barcode_gap < 10 or adapter_barcode_gap > 500:
        print('SKIPPING: read elements oddly arranged')
        return

    print('GOOD BARCODE {} TRAINING READ'.format(barcode_name))
    print('  base coords: adapter: {}-{} ({:.1f}%), barcode{}: {}-{} ({:.1f}%), '
          'ref: {}-{}'.format(adapter_start, adapter_end, adapter_identity, barcode_name,
                              barcode_start, barcode_end, barcode_identity, ref_start, ref_end))
    training_sample = get_training_sample_from_signal(signal, barcode_signal_start,
                                                      barcode_signal_end, signal_size)
    # TODO: print sample




def prep_native_read_end():
    pass


def prep_rapid_read_start():
    pass


def edlib_align(query_seq, ref_seq):
    alignment = edlib.align(query_seq, ref_seq, mode='HW', task='path')
    return (identity_from_edlib_cigar(alignment['cigar']),
            alignment['locations'][0][0], alignment['locations'][0][1])


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


def get_best_barcode(read_seq, barcode_seqs):
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
    if best_barcode_identity < MIN_BARCODE_IDENTITY:
        return 'none', best_barcode_identity, best_start, best_end
    if best_second_best_diff < MIN_BEST_SECOND_BEST_DIFF:
        return 'too close', best_barcode_identity, best_start, best_end
    else:
        return best_barcode_name, best_barcode_identity, best_start, best_end


def dtw_align(ref_signal, query_signal):

    distance, signal_start, signal_end = 0, 0, 0  # TEMP

    return distance, signal_start, signal_end


def get_best_barcode_dtw(read_signal, barcode_signals, start_trim_pos):
    best_barcode_name, best_barcode_distance = None, float('inf')
    best_start, best_end = 0, 0
    for barcode_name, barcode_signal in barcode_signals.items():
        distance, start, end, _ = semi_global_dtw_with_rescaling(read_signal, barcode_signal)
        start += start_trim_pos
        end += start_trim_pos
        print('  barcode{} DTW: {}-{} ({:.2f})'.format(barcode_name, start, end, distance))
        if distance < best_barcode_distance:
            best_barcode_distance = distance
            best_barcode_name = barcode_name
            best_start, best_end = start, end
    return best_barcode_name, best_start, best_end


def get_training_sample_from_signal(signal, include_start, include_end, signal_size):
    """
    This function takes in a large signal and returns a training-sized chunk which includes the
    specified range.
    """
    print('  choosing a training sample centred around: {}-{}'.format(include_start, include_end))
    include_size = include_end - include_start
    min_start = max(0, include_start + include_size - signal_size)
    training_start = int(round(random.uniform(min_start, include_start)))
    training_end = training_start + signal_size
    print('  training signal coords: {}-{}'.format(training_start, training_end))
    return signal[training_start:training_end]
