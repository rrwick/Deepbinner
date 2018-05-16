#!/usr/bin/env python3
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

import argparse
import collections
import statistics


def get_arguments():
    parser = argparse.ArgumentParser(description='Bin reads based on alignments to references',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     add_help=False)

    pos_args = parser.add_argument_group('Positional arguments (required)')
    pos_args.add_argument('paf_filename', type=str,
                          help='PAF file of read alignments to all references')
    pos_args.add_argument('read_lengths', type=str,
                          help='Tab-delimited file of read IDs and read lengths (make with: '
                               'tail -n+2 sequencing_summary.txt | cut -f2,13)')

    class_args = parser.add_argument_group('Read classification thresholds',
                                           'control how reads\' bases are assigned to references')
    class_args.add_argument('--min_align_len', type=int, required=False, default=50,
                            help='Alignments shorter than this are ignored')
    class_args.add_argument('--min_align_id', type=float, required=False, default=50.0,
                            help='Alignments with a percent identity lower than this are ignored')
    class_args.add_argument('--min_base_diff', type=float, required=False, default=5.0,
                            help='Read bases are not classified if the difference in percent '
                                 'identity between the best and second-best alignments is less '
                                 'than this')

    low_q_args = parser.add_argument_group('Low quality thresholds',
                                           'a read will be classified as low quality if it fails '
                                           'to meet both of these thresholds')
    low_q_args.add_argument('--low_q_bases', type=float, required=False, default=100,
                            help='At least this many bases aligned to a reference')
    low_q_args.add_argument('--low_q_percent', type=float, required=False, default=10,
                            help='At least this percent of the read aligned to a reference')

    chimera_args = parser.add_argument_group('Chimera thresholds',
                                             'a read will be classified as a chimera if it '
                                             'exceeds either of these thresholds')
    chimera_args.add_argument('--chimera_bases', type=float, required=False, default=50,
                              help='This many or more bases aligned to a secondary reference')
    chimera_args.add_argument('--chimera_percent', type=float, required=False, default=10,
                              help='This fraction or more of the aligned bases are to a '
                                   'secondary reference')

    help_args = parser.add_argument_group('Help')
    help_args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                           help='Show this help message and exit')

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    read_names, read_lengths = load_read_names_and_lengths(args.read_lengths)
    alignments, ref_names = load_alignments(args, read_names, read_lengths)
    ref_name_to_index = {b: a for a, b in enumerate(ref_names)}
    print_header(ref_names)

    for read_name in read_names:
        process_read(read_name, alignments, ref_names, read_lengths, ref_name_to_index, args)


def load_read_names_and_lengths(read_lengths_filename):
    read_names = set()
    read_lengths = {}
    with open(read_lengths_filename, 'rt') as read_lengths_file:
        for line in read_lengths_file:
            parts = line.strip().split('\t')
            read_names.add(parts[0])
            read_lengths[parts[0]] = int(parts[1])
    read_names = sorted(read_names)
    return read_names, read_lengths


def load_alignments(args, read_names, read_lengths):
    alignments = collections.defaultdict(list)
    ref_names = set()
    read_names = set(read_names)
    with open(args.paf_filename, 'rt') as paf:
        for line in paf:
            parts = line.strip().split('\t')
            read_name = parts[0]
            assert read_name in read_names
            ref_name = parts[5].split('__')[0]
            ref_names.add(ref_name)
            read_length = int(parts[1])
            assert read_lengths[read_name] == read_length
            identity = 100.0 * int(parts[9]) / int(parts[10])
            read_start = int(parts[2])
            read_end = int(parts[3])
            align_len = read_end - read_start
            if identity >= args.min_align_id and align_len >= args.min_align_len:
                alignments[read_name].append((ref_name, identity, read_start, read_end))
    ref_names = sorted(ref_names)
    return alignments, ref_names


def process_read(read_name, alignments, ref_names, read_lengths, ref_name_to_index, args):
    read_length = read_lengths[read_name]
    identity_per_base_per_ref = get_identities_per_ref(read_name, ref_names, read_length,
                                                       alignments)
    painted_read, read_identity = paint_read(identity_per_base_per_ref, read_length, ref_names,
                                             ref_name_to_index, args)
    matches, matched_bases, unmatched_bases, matched_percent = \
        quantify_matches(painted_read, read_length, ref_names, ref_name_to_index)

    best_match, second_best_match = sorted(matches.values(), reverse=True)[0:2]

    if read_length == 0:
        call = 'not basecalled'
    elif matched_bases < args.low_q_bases and matched_percent < args.low_q_percent:
        call = 'low quality'
    elif second_best_match >= args.chimera_bases:
        call = 'chimera'
    else:
        call = {v: k for k, v in matches.items()}[best_match]

    print_output_line(read_name, read_length, matched_bases, unmatched_bases, read_identity,
                      matches, ref_names, call)


def get_identities_per_ref(read_name, ref_names, read_length, alignments):
    identity_per_base_per_ref = {}
    for ref in ref_names:
        identity_per_base_per_ref[ref] = [0.0] * read_length
    if read_name in alignments:
        for ref, identity, read_start, read_end in alignments[read_name]:
            for i in range(read_start, read_end):
                if identity > identity_per_base_per_ref[ref][i]:
                    identity_per_base_per_ref[ref][i] = identity
    return identity_per_base_per_ref


def paint_read(identity_per_base_per_ref, read_length, ref_names, ref_name_to_index, args):
    painted_read = [None] * read_length
    used_identities = []
    for i in range(read_length):
        identities = {ref: identity_per_base_per_ref[ref][i] for ref in ref_names}
        best, second_best = sorted(identities.values(), reverse=True)[0:2]
        if best > 0.0 and best - second_best >= args.min_base_diff:
            ref_name = {v: k for k, v in identities.items()}[best]
            painted_read[i] = ref_name_to_index[ref_name]
            used_identities.append(best)
    if used_identities:
        read_identity = statistics.mean(used_identities)
    else:
        read_identity = 0.0
    return painted_read, read_identity


def quantify_matches(painted_read, read_length, ref_names, ref_name_to_index):
    # noinspection PyArgumentList
    counts = collections.Counter(painted_read)
    if None in counts:
        unmatched_bases = counts[None]
        matched_bases = read_length - counts[None]
    else:
        unmatched_bases = 0
        matched_bases = read_length
    if read_length > 0:
        matched_percent = 100.0 * matched_bases / read_length
    else:
        matched_percent = 0.0

    matches = {}
    for ref in ref_names:
        ref_index = ref_name_to_index[ref]
        if ref_index in counts:
            matches[ref] = counts[ref_index]
        else:
            matches[ref] = 0.0

    if None in counts:
        matches['none'] = counts[None] / read_length
    else:
        matches['none'] = 0.0

    return matches, matched_bases, unmatched_bases, matched_percent


def print_header(ref_names):
    header = ['Read_name', 'Read_length', 'Read_identity', 'Unaligned_bases', 'Aligned_bases'] + \
             ref_names + ['Classification']
    print('\t'.join(header))


def print_output_line(read_name, read_length, matched_bases, unmatched_bases, read_identity,
                      matches, ref_names, call):
    output_line = [read_name, str(read_length), '%.1f' % read_identity, str(unmatched_bases),
                   str(matched_bases)]
    output_line += [str(matches[r]) if matches[r] > 0 else '' for r in ref_names]
    output_line += [call]
    print('\t'.join(output_line))


if __name__ == '__main__':
    main()
