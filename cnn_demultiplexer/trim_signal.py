
import numpy as np


initial_trim_size = 10
trim_increment = 25
stdev_threshold = 20

look_forward_windows = 4
window_count_threshold = 3


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

    print(start_trim_pos, end_trim_pos)
    print(start_trim_pos, end_trim_pos)

    # Plot the resulting signal (for debugging)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(start_signal)
        if good_start:
            plt.axvspan(start_trim_pos, start_trim_pos + signal_size, alpha=0.2, color='red')
        plt.show()

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
    # Always trim off the first few bases as these are often dodgy.
    pos = initial_trim_size

    # Look at the stdev of the signal in the upcoming windows. Trimming is finished when:
    #  1. the next window has a high stdev
    #  2. enough of the other upcoming windows have a high stdev
    while True:
        next_window_stdev = get_window_stdev(signal, pos, 0)
        if next_window_stdev > stdev_threshold:
            upcoming_window_stdevs = [get_window_stdev(signal, pos, i)
                                      for i in range(look_forward_windows)]
            num_high_stdevs = sum(1 if x > stdev_threshold else 0
                                  for x in upcoming_window_stdevs)
            if num_high_stdevs >= window_count_threshold:
                return pos
        pos += trim_increment


def get_window_stdev(signal, current_pos, window_num):
    window_start = current_pos + (window_num * trim_increment)
    window_end = window_start + trim_increment
    if window_end > len(signal):
        raise CannotTrim
    return np.std(signal[window_start:window_end])


def find_signal_end_pos(signal):
    """
    This does the same thing as find_signal_start_pos, but for the end of the read.
    """
    return len(signal) - find_signal_start_pos(signal[::-1])
