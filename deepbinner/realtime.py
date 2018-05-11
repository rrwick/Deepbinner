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
import shutil
import sys
import time

from .classify import load_and_check_models, classify_fast5_files, set_tensorflow_threads
from .bin import class_to_class_names


def realtime(args):
    print()
    args.verbose = False
    nested_out_dir = pathlib.Path(args.in_dir) in pathlib.Path(args.out_dir).parents

    set_tensorflow_threads(args)
    start_model, start_input_size, end_model, end_input_size, output_size, model_count = \
        load_and_check_models(args.start_model, args.end_model, args.scan_size,
                              out_dest=sys.stdout)

    make_output_dir(args.out_dir)
    try:
        waiting = False
        while True:
            fast5s = look_for_new_fast5s(args.in_dir, args.out_dir, nested_out_dir)
            if fast5s:
                if waiting:
                    print()
                time.sleep(5)  # wait a bit to make sure any file moves are finished
                classify_and_move(fast5s, args, start_model, start_input_size, end_model,
                                  end_input_size, output_size)
                waiting = False
            else:
                if waiting:
                    print('.', end='', flush=True)
                else:
                    print('\nWaiting for new fast5 files (press Ctrl-C to stop)', end='',
                          flush=True)
                    waiting = True
            time.sleep(5)
    except KeyboardInterrupt:
        print('\n\nStopping Deepbinner real-time binning\n')


def look_for_new_fast5s(in_dir, out_dir, nested_out_dir):
    in_dir_fast5s = [str(x) for x in sorted(pathlib.Path(in_dir).glob('**/*.fast5'))]
    if nested_out_dir:
        out_dir_fast5s = set(str(x) for x in sorted(pathlib.Path(out_dir).glob('**/*.fast5')))
        in_dir_fast5s = [x for x in in_dir_fast5s if x not in out_dir_fast5s]
    return in_dir_fast5s


def classify_and_move(fast5s, args, start_model, start_input_size, end_model, end_input_size,
                      output_size):
    print()
    print('Found {:,} new fast5 files'.format(len(fast5s)))
    classifications, read_id_to_fast5_file = \
        classify_fast5_files(fast5s, start_model, start_input_size, end_model, end_input_size,
                             output_size, args, full_output=False)
    counts = collections.defaultdict(int)
    for read_id, barcode_call in classifications.items():
        fast5_file = read_id_to_fast5_file[read_id]
        out_dir = pathlib.Path(args.out_dir) / class_to_class_names(barcode_call)
        if not out_dir.is_dir():
            try:
                os.makedirs(str(out_dir))
            except (FileNotFoundError, OSError, PermissionError):
                sys.exit('Error: unable to create output directory {}'.format(out_dir))
            try:
                if args.copy:
                    shutil.copy(fast5_file, out_dir)
                else:
                    shutil.move(fast5_file, out_dir)
            except (FileNotFoundError, OSError, PermissionError):
                print('Error: could not move {} to {}'.format(fast5_file, out_dir))
        counts[barcode_call] += 1
    print()


def make_output_dir(out_dir):
    if pathlib.Path(out_dir).is_file():
        sys.exit('Error: {} is an existing file'.format(out_dir))
    if not pathlib.Path(out_dir).is_dir():
        try:
            os.makedirs(out_dir, exist_ok=True)
            print()
            print('Making output directory: {}/'.format(out_dir))
        except (FileNotFoundError, OSError, PermissionError):
            sys.exit('Error: unable to create output directory {}'.format(out_dir))
