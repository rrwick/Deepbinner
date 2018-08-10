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

    if args.sequencing_summary is not None:
        albacore_barcodes = load_albacore_barcodes_from_sequencing_summary(args.sequencing_summary)
    else:
        albacore_barcodes = None

    # For the ligation kit we need to align to reference (but not for the rapid kit).
    if args.kit == 'EXP-NBD103_start' or args.kit == 'EXP-NBD103_end':
        mappy_aligner = mp.Aligner(args.ref_fasta)
    else:
        mappy_aligner = None

    for fast5_file in fast5s:
        read_id, signal = get_read_id_and_signal(fast5_file)
        if read_id not in read_seqs:
            continue

        print('', file=sys.stderr)
        print(fast5_file, file=sys.stderr)
        print('  read ID: {}'.format(read_id), file=sys.stderr)

        if args.kit == 'EXP-NBD103_start':
            prep_native_read_start(signal, read_seqs[read_id], mappy_aligner, args.signal_size)

        if args.kit == 'EXP-NBD103_end':
            prep_native_read_end(signal, read_seqs[read_id], mappy_aligner, args.signal_size)

        elif args.kit == 'SQK-RBK004_start':
            prep_rapid_read_start()


def load_albacore_barcodes_from_sequencing_summary(sequence_summary_filename):
    return None  # TEMP, TODO
