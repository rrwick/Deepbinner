
import numpy as np

# To determine the start of the real signal (as opposed to the open pore signal), we look at the
# median absolute deviation over a sliding window and note when it exceeds a threshold.
sliding_window_size = 250
mad_threshold = 20


def clean_signal(start_signal, end_signal, signal_size, plot):
    """
    This function takes signals as input (in the form of a comma-delimited string) and it trims
    them down to the signal size, removing open pore signal from the ends.
    """
    start_signal = [int(x) for x in start_signal.split(',')]
    end_signal = [int(x) for x in end_signal.split(',')]

    good_start, good_end = True, True

    start_trim_pos = find_signal_start_pos(start_signal)
    if start_trim_pos == 0:
        good_start = False
    trimmed_start = start_signal[start_trim_pos:start_trim_pos + signal_size]
    if len(start_signal) < signal_size:
        good_start = False

    end_trim_pos = find_signal_end_pos(end_signal)
    if end_trim_pos == 0:
        good_end = False
    trimmed_end = end_signal[end_trim_pos - signal_size:end_trim_pos]
    if len(end_signal) < signal_size:
        good_end = False

    print(start_trim_pos, end_trim_pos)

    # Plot the resulting signal (for debugging)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(start_signal)
        plt.axvline(x=start_trim_pos, color='red')
        plt.axvline(x=start_trim_pos + signal_size, color='red')
        plt.show()

        plt.plot(end_signal)
        plt.axvline(x=end_trim_pos, color='red')
        plt.axvline(x=end_trim_pos - signal_size, color='red')
        plt.show()

    trimmed_start = ','.join(str(x) for x in trimmed_start)
    trimmed_end = ','.join(str(x) for x in trimmed_end)

    return trimmed_start, trimmed_end, good_start, good_end


def find_signal_start_pos(signal):
    """
    Given a signal, this function attempts to identify the approximate position where the open
    pore signal ends and the real signal begins.
    """
    median_start = int(np.median(signal[:50]))
    for i in range(-(sliding_window_size // 2), len(signal) - sliding_window_size):

        # Grab signal in the window. If the window overlaps the front of the signal, pad it with
        # The average signal start value
        if i >= 0:
            window = signal[i:i + sliding_window_size]
            pad_size = 0
        else:
            window = signal[0:i + sliding_window_size]
            pad_size = sliding_window_size - len(window)
            window = ([median_start] * pad_size) + window

        if median_absolute_deviation(window) > mad_threshold:
            return i + (sliding_window_size // 2)
    return 0


def find_signal_end_pos(signal):
    """
    This does the same thing as find_signal_start_pos, but for the end of the read.
    """
    return len(signal) - find_signal_start_pos(signal[::-1])


def median_absolute_deviation(arr):
    med = np.median(arr)
    return np.median(np.abs(arr - med))
