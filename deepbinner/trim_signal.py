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


def clean_signal(start_signal, end_signal, signal_size, plot):
    """
    This function takes signals as input (in the form of a comma-delimited string) and it trims
    them down to the signal size, removing open pore signal from the ends.
    """
    start_signal = [int(x) for x in start_signal.split(',')]
    end_signal = [int(x) for x in end_signal.split(',')]

    good_start, good_end = True, True

    try:
        start_trim_pos = find_signal_start_pos(start_signal)
    except CannotTrim:
        start_trim_pos = 0
        good_start = False
    trimmed_start = start_signal[start_trim_pos:start_trim_pos + signal_size]
    if len(trimmed_start) < signal_size:
        good_start = False

    try:
        end_trim_pos = find_signal_end_pos(end_signal)
    except CannotTrim:
        end_trim_pos = 0
        good_end = False
    trimmed_end = end_signal[end_trim_pos - signal_size:end_trim_pos]
    if len(trimmed_end) < signal_size:
        good_end = False

    # Plot the resulting signal (for debugging)
    if plot:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 5))

        fig.add_subplot(2, 1, 1)
        plt.plot(start_signal)
        if good_start:
            plt.axvspan(start_trim_pos, start_trim_pos + signal_size, alpha=0.2, color='red')

        fig.add_subplot(2, 1, 2)
        plt.plot(end_signal)
        if good_end:
            plt.axvspan(end_trim_pos - signal_size, end_trim_pos, alpha=0.2, color='red')

        plt.show()

    trimmed_start = ','.join(str(x) for x in trimmed_start)
    trimmed_end = ','.join(str(x) for x in trimmed_end)

    return trimmed_start, trimmed_end, good_start, good_end


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

    # Always trim off the first few bases as these are often dodgy.
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


def find_signal_end_pos(signal):
    """
    This does the same thing as find_signal_start_pos, but for the end of the read.
    """
    return len(signal) - find_signal_start_pos(signal[::-1])


def get_window_stdev(signal, current_pos, window_num, increment):
    window_start = current_pos + (window_num * increment)
    window_end = window_start + increment
    if window_end > len(signal):
        raise CannotTrim
    return np.std(signal[window_start:window_end])


def too_much_open_pore(signal):
    """
    This function returns True if too much of the signal has a low standard deviation.
    """
    window_size = 100
    stdev_threshold = 30
    fraction = 0.4

    window_count = int(len(signal) / window_size)
    window_stdevs = [get_window_stdev(signal, 0, i, window_size) for i in range(window_count)]
    num_low_stdevs = sum(1 if x < stdev_threshold else 0 for x in window_stdevs)
    return num_low_stdevs / window_count > fraction


def normalise(signal):
    if len(signal) == 0:
        return signal
    mean = np.mean(signal)
    stdev = np.std(signal)
    if stdev > 0.0:
        return (signal - mean) / stdev
    else:
        return signal - mean
