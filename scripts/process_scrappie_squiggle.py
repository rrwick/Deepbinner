#!/usr/bin/env python3
"""
Copyright 2018 Ryan Wick (rrwick@gmail.com)
https://github.com/rrwick/Deepbinner/

This script is used to convert the output of Scrappie squiggle into a signal I can DTW align to
real read signals. It was used to make the signal values in the signals.py file.

This file is part of Deepbinner. Deepbinner is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version. Deepbinner is distributed
in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along with Deepbinner.
If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import numpy as np


def main():
    time = 0.0
    squiggles = {}
    current_seq = None
    with open(sys.argv[1], 'rt') as scrappie_squiggle_output:
        for line in scrappie_squiggle_output:

            # A hash indicates the start of a new sequence.
            if line.startswith('#'):
                current_seq = line.strip()[1:]
                squiggles[current_seq] = []
                time = 0.0
                continue

            parts = line.rstrip().split('\t')
            if parts[0] == 'pos':
                continue
            current = float(parts[2])
            dwell = float(parts[4])
            squiggles[current_seq].append((time + 0.01, current))
            time += dwell
            squiggles[current_seq].append((time - 0.01, current))

    for seq_name, squiggle in squiggles.items():
        x_vals = [s[0] for s in squiggle]
        y_vals = [s[1] for s in squiggle]
        new_y = linear_resample(x_vals, y_vals)
        print(seq_name, end='\t')
        print(', '.join(['{:.3f}'.format(y) for y in new_y]))


def linear_resample(x_vals, y_vals):
    max_x = int(round(x_vals[-1]))
    x_vals = np.array(x_vals)
    new_x = list(range(max_x + 1))
    new_y = []
    for x in new_x:
        i = (np.abs(x_vals - x)).argmin()
        new_y.append(y_vals[i])
    return new_y


if __name__ == '__main__':
    main()
