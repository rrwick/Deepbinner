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

import mappy as mp
import pathlib
import sys

from .load_fast5s import get_read_id_and_signal, find_all_fast5s
from .misc import load_fastq
from .prep_native_start import prep_native_read_start
from .prep_native_end import prep_native_read_end
from .prep_rapid_start import prep_rapid_read_start


def prep(args):
    if pathlib.Path(args.fast5_dir).is_dir():
        fast5s = find_all_fast5s(args.fast5_dir)
    else:
        fast5s = [args.fast5_dir]

    read_seqs = load_fastq(args.fastq)

    albacore_barcodes = load_albacore_barcodes_from_sequencing_summary(args.sequencing_summary)

    # For the ligation kit we need to align to reference (but not for the rapid kit).
    if args.kit == 'EXP-NBD103_start' or args.kit == 'EXP-NBD103_end':
        mappy_aligner = mp.Aligner(args.ref)
    else:
        mappy_aligner = None

    read_count = 0
    for fast5_file in fast5s:
        read_id, signal = get_read_id_and_signal(fast5_file)
        if read_id not in read_seqs:
            continue

        print('', file=sys.stderr)
        print(fast5_file, file=sys.stderr)
        print('  read ID: {}'.format(read_id), file=sys.stderr)

        if albacore_barcodes is not None:
            try:
                albacore_barcode = albacore_barcodes[read_id]
            except KeyError:
                albacore_barcode = None
        else:
            albacore_barcode = None

        if args.kit == 'EXP-NBD103_start':
            prep_native_read_start(signal, read_seqs[read_id], mappy_aligner, args.signal_size,
                                   albacore_barcode)

        if args.kit == 'EXP-NBD103_end':
            prep_native_read_end(signal, read_seqs[read_id], mappy_aligner, args.signal_size,
                                 albacore_barcode)

        elif args.kit == 'SQK-RBK004_start':
            prep_rapid_read_start()

        read_count += 1
        if args.read_limit is not None:
            if read_count >= args.read_limit:
                break

    print('', file=sys.stderr)


def load_albacore_barcodes_from_sequencing_summary(sequence_summary_filename):
    if sequence_summary_filename is None:
        return None
    albacore_barcodes = {}
    read_id_column, barcode_column = None, None
    with open(sequence_summary_filename, 'rt') as sequence_summary_file:
        for line in sequence_summary_file:
            parts = line.strip().split('\t')
            if read_id_column is None:  # if this is the first line looked at
                try:
                    read_id_column = parts.index('read_id')
                except ValueError:
                    sys.exit('Error: {} does not have a read_id '
                             'column'.format(sequence_summary_filename))
                try:
                    barcode_column = parts.index('barcode_arrangement')
                except ValueError:
                    sys.exit('Error: {} does not have a barcode_arrangement '
                             'column'.format(sequence_summary_filename))
            if parts[read_id_column] != 'read_id':  # if this isn't a header line
                read_id = parts[read_id_column]
                barcode = parts[barcode_column].replace('barcode', '')
                albacore_barcodes[read_id] = barcode
    return albacore_barcodes
