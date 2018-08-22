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
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys

from .misc import get_open_function


def bin_reads(args):
    classifications = load_classifications(args.classes)
    class_names = sorted(class_to_class_names(x) for x in set(classifications.values()))
    input_type = get_sequence_file_type(args.reads)
    out_filenames = get_output_filenames(class_names, args.out_dir, input_type)
    make_output_dir(args.out_dir, out_filenames)
    write_read_files(args.reads, classifications, out_filenames, input_type)


def load_classifications(class_filename):
    print('\nLoading classifications...', end='', flush=True)
    if not pathlib.Path(class_filename).is_file():
        sys.exit('Error: {} does not exist'.format(class_filename))

    read_id_warning = False
    classifications = {}
    with open(class_filename, 'rt') as class_file:
        for line in class_file:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            if parts[0].lower() == 'read_id':
                continue
            read_id, classification = parts[0:2]
            if len(read_id) != 36:
                read_id_warning = True
            elif not(read_id[8] == read_id[13] == read_id[18] == read_id[23] == '-'):
                read_id_warning = True
            if classification == 'none':
                classification = None
            else:
                try:
                    classification = int(classification)
                except ValueError:
                    sys.exit('Error: read {} has a non-integer bin of {}'.format(read_id,
                                                                                 classification))
            classifications[read_id] = classification

    print(' done')
    if read_id_warning:
        print('Warning: one or more read IDs did not conform to the expected format (UUID)')

    print('{:,} total classifications found'.format(len(classifications)))
    print()
    return classifications


def make_output_dir(out_dir, out_filenames):
    if pathlib.Path(out_dir).is_file():
        sys.exit('Error: {} is an existing file'.format(out_dir))
    if not pathlib.Path(out_dir).is_dir():
        try:
            os.makedirs(out_dir, exist_ok=True)
            print('Making output directory: {}/'.format(out_dir))
        except (FileNotFoundError, OSError, PermissionError):
            sys.exit('Error: unable to create output directory {}'.format(out_dir))
    for out_filename in out_filenames.values():
        if pathlib.Path(out_filename).exists():
            sys.exit('Error: {} already exists'.format(out_filename))
        if pathlib.Path(out_filename + '.gz').exists():
            sys.exit('Error: {}.gz already exists'.format(out_filename))
    print()


def class_to_class_names(classification):
    if classification is None:
        return 'unclassified'
    elif isinstance(classification, str):
        return classification
    else:
        return 'barcode{:02d}'.format(classification)


def get_output_filenames(class_names, out_dir, input_type):
    output_filenames = collections.OrderedDict()
    for class_name in class_names:
        class_filename = class_name + '.' + input_type
        output_filenames[class_name] = str(pathlib.Path(out_dir) / class_filename)
    return output_filenames


def write_read_files(reads_filename, classifications, out_filenames, input_type):
    p = re.compile(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}')

    open_func = get_open_function(reads_filename)
    bin_counts = collections.defaultdict(int)

    out_files = {}
    for class_name, out_file in out_filenames.items():
        out_files[class_name] = open(out_file, 'wt')

    count, interval = 0, random.randint(90, 110)
    with open_func(reads_filename, 'rt') as reads:
        for line in reads:
            if count % interval == 0:
                print_progress(count)
                interval = random.randint(90, 110)
            count += 1

            read_line_1 = line
            read_line_2 = next(reads)
            if input_type == 'fastq':
                read_line_3 = next(reads)
                read_line_4 = next(reads)
            else:
                read_line_3, read_line_4 = '', ''

            try:
                read_id = p.search(read_line_1).group(0)
            except AttributeError:
                sys.exit('Error: could not find read ID in header: {}'.format(read_line_1))

            try:
                read_class = classifications[read_id]
            except KeyError:
                read_class = 'not found'

            class_name = class_to_class_names(read_class)
            bin_counts[class_name] += 1

            out_files[class_name].write(read_line_1)
            out_files[class_name].write(read_line_2)
            out_files[class_name].write(read_line_3)
            out_files[class_name].write(read_line_4)

    print_progress(count, carriage_return=False)
    print('\n')

    for class_name in out_filenames:
        out_files[class_name].close()

    print_summary_and_zip(bin_counts, out_filenames)


def print_progress(count, carriage_return=True):
    print('Writing reads: {:,}'.format(count) + ' ', end='')
    if carriage_return:
        print('\r', end='')


def get_sequence_file_type(filename):
    if not pathlib.Path(filename).is_file():
        sys.exit('Error: could not find ' + filename)

    open_func = get_open_function(filename)
    with open_func(filename, 'rt') as seq_file:
        try:
            first_char = seq_file.read(1)
        except UnicodeDecodeError:
            first_char = ''

    if first_char == '>':
        return 'fasta'
    elif first_char == '@':
        return 'fastq'
    else:
        raise ValueError('Error: could not determine file format (should be fasta or fastq)')


def print_summary_and_zip(bin_counts, out_filenames):
    if shutil.which('pigz'):
        gzip = 'pigz'
        print('Gzipping reads (with pigz):')
    else:
        gzip = 'gzip'
        print('Gzipping reads:')
    print('  Barcode       Reads     File')
    class_names = out_filenames.keys()
    if 'not found' in bin_counts:
        class_names.append('not found')
    for class_name in class_names:
        if class_name in out_filenames:
            filename = out_filenames[class_name]
            subprocess.check_output([gzip, filename])
            gzipped_filename = filename + '.gz'
        else:
            gzipped_filename = ''
        display_name = 'none' if class_name == 'unclassified' else class_name
        print('  {:<9} {:>9}     {}'.format(display_name, bin_counts[class_name], gzipped_filename))
    print()
