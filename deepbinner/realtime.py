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
from .misc import print_summary_table

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
                time.sleep(5)  # wait a bit to make sure any file moves are finished
                classify_and_move(fast5s, args, start_model, start_input_size, end_model,
                                  end_input_size, output_size)
                waiting = False
            else:
                if waiting:
                    print('.', end='', flush=True)
                else:
                    print('\nWaiting for new fast5 files (Ctrl-C to stop)', end='',
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
    print('Found {:,} fast5 files in {}'.format(len(fast5s), args.in_dir))

    # We don't classify all found fast5s, because if there are a ton it will take a long time to
    # finish, and nothing will be moved while we wait. Instead we grab the first
    fast5s = fast5s[:args.batch_size * 10]

    classifications, read_id_to_fast5_file = \
        classify_fast5_files(fast5s, start_model, start_input_size, end_model, end_input_size,
                             output_size, args, full_output=False)
    print()
    move_classified_fast5s(classifications, read_id_to_fast5_file, args, fast5s)
    print_summary_table(classifications, output=sys.stdout)


def move_classified_fast5s(classifications, read_id_to_fast5_file, args, fast5s):
    move_count, fail_move_already_exists, fail_move_other_reason = 0, 0, 0
    counts = collections.defaultdict(int)
    for read_id, barcode_call in classifications.items():
        fast5_file = read_id_to_fast5_file[read_id]

        out_dir = pathlib.Path(args.out_dir) / class_to_class_names(barcode_call)
        if not out_dir.is_dir():
            try:
                os.makedirs(str(out_dir))
            except (FileNotFoundError, OSError, PermissionError):
                sys.exit('Error: unable to create output directory {}'.format(out_dir))

        dest_filepath = out_dir / pathlib.Path(fast5_file).name
        if dest_filepath.is_file():
            fail_move_already_exists += 1
        else:
            try:
                shutil.move(fast5_file, out_dir)
                move_count += 1
            except (FileNotFoundError, OSError, PermissionError):
                fail_move_other_reason += 1

        counts[barcode_call] += 1
        print_moving_progress(move_count, len(fast5s))

    print()
    print_moving_error_messages(fail_move_already_exists, fail_move_other_reason, args.out_dir)

    # If we couldn't move any files, then something is wrong and we should quit.
    if move_count == 0:
        sys.exit('Error: no files were successfully moved to {}'.format(args.out_dir))


def print_moving_error_messages(already_exists, other_reason, out_dir):
    if already_exists == 1:
        print('Error: could not move 1 fast5 file because it already exists in {}'.format(out_dir))
    elif already_exists > 1:
        print('Error: could not move {} fast5 files because they already exist '
              'in {}'.format(already_exists, out_dir))
    if other_reason == 1:
        print('Error: failed to move 1 fast5 file to {}'.format(out_dir))
    elif other_reason > 1:
        print('Error: failed to move {} fast5 files to {}'.format(other_reason, out_dir))


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


def print_moving_progress(completed, total):
    percent = 100.0 * completed / total
    print('\rMoving fast5s:      {} / {} ({:.1f}%)'.format(completed, total, percent),
          end='', flush=True)
