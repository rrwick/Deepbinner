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

import sys


def refine_training_samples(args):
    print('\nRefining training data', file=sys.stderr)

    with open(args.training_data) as old_training_data, open(args.classification_data) as classes:
        _ = next(classes)  # Skip the header line

        match_count, total_count = 0, 0
        for training_line, class_line in zip(old_training_data, classes):
            classes_parts = class_line.rstrip().split('\t')
            class_barcode = classes_parts[1]
            if class_barcode == 'none':
                class_barcode = 0
            else:
                class_barcode = int(class_barcode)

            train_barcode = training_line.split('\t')[0]
            # Sanity check to make sure the two files' lines are aligned.
            assert classes_parts[0].split('_')[-1] == train_barcode
            train_barcode = int(train_barcode)

            if train_barcode == class_barcode:
                print(training_line, end='')
                match_count += 1
            total_count += 1

            print('\rMatches: {} / {} ({:.2f}%)'.format(match_count, total_count,
                                                        100.0 * match_count / total_count),
                  file=sys.stderr, end='')

        print('\n', file=sys.stderr)
