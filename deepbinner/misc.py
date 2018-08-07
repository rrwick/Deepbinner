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
import gzip
import sys


def print_summary_table(classifications, output=sys.stderr):
    counts = collections.defaultdict(int)
    for barcode in classifications.values():
        counts[barcode] += 1
    print('', file=output)
    print('Barcode     Count', file=output)

    number_barcodes, string_barcodes = [], []
    for barcode in counts.keys():
        try:
            number_barcodes.append(int(barcode))
        except ValueError:
            string_barcodes.append(barcode)
    sorted_barcodes = sorted(number_barcodes) + sorted(string_barcodes)

    for barcode in sorted_barcodes:
        print('{:>7} {:>9}'.format(barcode, counts[str(barcode)]), file=output)
    print('', file=output)


def get_compression_type(filename):
    """
    Attempts to guess the compression (if any) on a file using the first few bytes.
    http://stackoverflow.com/questions/13044562
    """
    magic_dict = {'gz': (b'\x1f', b'\x8b', b'\x08'),
                  'bz2': (b'\x42', b'\x5a', b'\x68'),
                  'zip': (b'\x50', b'\x4b', b'\x03', b'\x04')}
    max_len = max(len(x) for x in magic_dict)
    with open(filename, 'rb') as unknown_file:
        file_start = unknown_file.read(max_len)
    compression_type = 'plain'
    for file_type, magic_bytes in magic_dict.items():
        if file_start.startswith(magic_bytes):
            compression_type = file_type
    if compression_type == 'bz2':
        sys.exit('Error: cannot use bzip2 format - use gzip instead')
    if compression_type == 'zip':
        sys.exit('Error: cannot use zip format - use gzip instead')
    return compression_type


def get_open_function(filename):
    """
    Returns either open or gzip.open, as appropriate for the file.
    """
    if get_compression_type(filename) == 'gz':
        return gzip.open
    else:  # plain text
        return open


def load_fastq(fastq_filename):
    """
    Returns a list of tuples (header, seq, qual) for each record in the fastq file.
    """
    reads = {}
    with get_open_function(fastq_filename)(fastq_filename, 'rt') as fastq:
        for line in fastq:
            full_name = line.strip()[1:]
            short_name = full_name.split()[0]
            sequence = next(fastq).strip()
            next(fastq)
            next(fastq)
            reads[short_name] = sequence
    return reads
