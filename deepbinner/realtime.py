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

import os
import pathlib
import sys
import time

from .classify import load_and_check_models, classify_fast5_files


def realtime(args):
    start_model, start_input_size, end_model, end_input_size, output_size, model_count = \
        load_and_check_models(args.start_model, args.end_model, args.scan_size)

    make_output_dir(args.out_dir)
    try:
        waiting = False
        while True:
            fast5s = look_for_new_fast5s(args.in_dir, args.out_dir)
            if fast5s:
                classify_and_move(fast5s, args, start_model, start_input_size, end_model,
                                  end_input_size, output_size)
                waiting = False
            else:
                if waiting:
                    print('.', end='', flush=True)
                else:
                    print('Waiting for new fast5 files (press Ctrl-C to stop)', end='', flush=True)
                    waiting = True
            time.sleep(5)
    except KeyboardInterrupt:
        print('\n\nStopping on user request\n')


def look_for_new_fast5s(in_dir, out_dir):
    return []  # TEMP


def classify_and_move(fast5s, args, start_model, start_input_size, end_model, end_input_size,
                      output_size):
    print('\n')
    print('Found {:,} new fast5 files'.format(len(fast5s)))
    classifications, read_id_to_fast5_file = \
        classify_fast5_files(fast5s, start_model, start_input_size, end_model, end_input_size,
                             output_size, args)





def make_output_dir(out_dir):
    print()
    if pathlib.Path(out_dir).is_file():
        sys.exit('Error: {} is an existing file'.format(out_dir))
    if not pathlib.Path(out_dir).is_dir():
        try:
            os.makedirs(out_dir, exist_ok=True)
            print('Making output directory: {}/'.format(out_dir))
        except (FileNotFoundError, OSError, PermissionError):
            sys.exit('Error: unable to create output directory {}'.format(out_dir))
    print()
