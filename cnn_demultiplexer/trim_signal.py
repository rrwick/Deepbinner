
import numpy

# To determine the start of the real signal (as opposed to the open pore signal), we look at the
# median absolute deviation over a sliding window and note when it exceeds a threshold.
sliding_window_size = 250
mad_threshold = 20


def find_signal_start_pos(signal):
    """
    Given a signal, this function attempts to identify the approximate position where the open
    pore signal ends and the real signal begins.
    """
    for i in range(len(signal) - sliding_window_size):
        if median_absolute_deviation(signal[i:i+sliding_window_size]) > mad_threshold:
            return i + (sliding_window_size // 2)
    return 0


def median_absolute_deviation(arr):
    med = np.median(arr)
    return np.median(np.abs(arr - med))
