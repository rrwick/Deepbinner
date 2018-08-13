"""
Copyright 2018 Ryan Wick (rrwick@gmail.com)
https://github.com/rrwick/Deepbinner/

This file takes care of trimming read signals to remove open pore signal before/after a read's
start/end. It's all fairly ad hoc and could probably use some more attention later.

This file is part of Deepbinner. Deepbinner is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version. Deepbinner is distributed
in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along with Deepbinner.
If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np


class CannotTrim(IndexError):
    pass


def find_signal_start_pos(signal):
    """
    Given a signal, this function attempts to identify the approximate position where the open
    pore signal ends and the real signal begins.
    """
    initial_trim_size = 10
    trim_increment = 25
    stdev_threshold = 20
    look_forward_windows = 5
    window_count_threshold = 4

    # Always trim off the first few values as these are often dodgy.
    pos = initial_trim_size

    # Look at the stdev of the signal in the upcoming windows. Trimming is finished when:
    #  1. the next window has a high stdev
    #  2. enough of the other upcoming windows have a high stdev
    while True:
        next_window_stdev = get_window_stdev(signal, pos, 0, trim_increment)
        if next_window_stdev > stdev_threshold:
            upcoming_window_stdevs = [get_window_stdev(signal, pos, i, trim_increment)
                                      for i in range(look_forward_windows)]
            num_high_stdevs = sum(1 if x > stdev_threshold else 0
                                  for x in upcoming_window_stdevs)
            if num_high_stdevs >= window_count_threshold:
                return pos
        pos += trim_increment


def get_window_stdev(signal, current_pos, window_num, increment):
    window_start = current_pos + (window_num * increment)
    window_end = window_start + increment
    if window_end > len(signal):
        raise CannotTrim
    return np.std(signal[window_start:window_end])


def normalise(signal):
    if len(signal) == 0:
        return signal
    mean = np.mean(signal)
    stdev = np.std(signal)
    if stdev > 0.0:
        return (signal - mean) / stdev
    else:
        return signal - mean
