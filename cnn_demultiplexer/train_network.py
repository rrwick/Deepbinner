
import sys
import os
import h5py
import random
import statistics

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Dropout, Flatten
from keras.models import Model, Sequential


# Parameters
signal_length = 1000
barcode_count = 12
test_fraction = 0.1
epochs = 100
optimizer = 'rmsprop'
loss = 'binary_crossentropy'
activation = 'relu'

# To determine the start of the real signal (as opposed to the open pore signal), we look at the
# median absolute deviation over a sliding window and note when it exceeds a threshold.
sliding_window_size = 250
mad_threshold = 20


def main():
    training_data_filename = sys.argv[1]

    print()
    training_signals, training_labels, testing_signals, testing_labels = \
        load_train_and_test_sets(training_data_filename)

    model = Sequential()

    model.add(Conv1D(filters=16, kernel_size=3, activation=activation, padding='same', input_shape=(signal_length,1)))
    model.add(Conv1D(filters=16, kernel_size=3, activation=activation))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    
    model.add(Conv1D(filters=32, kernel_size=3, activation=activation, padding='same'))
    model.add(Conv1D(filters=32, kernel_size=3, activation=activation))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    
    model.add(Conv1D(filters=32, kernel_size=3, activation=activation, padding='same'))
    model.add(Conv1D(filters=32, kernel_size=3, activation=activation))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(barcode_count + 1, activation='softmax'))



    for layer in model.layers:
        print()
        print(layer.input)
        print(layer.output)
        print(layer.get_config())
        print()

    quit()


    model.compile(optimizer=optimizer, loss=loss)

    try:
        model.fit(training_signals, training_labels,
                  epochs=epochs,
                  batch_size=256,
                  shuffle=True,
                  validation_data=(testing_signals, testing_labels))
    except KeyboardInterrupt:
        pass

    quit()





def load_train_and_test_sets(training_data_filename):
    random.seed(0)

    training_data, testing_data = [], []

    print('Loading data from file... ', end='')
    with open(training_data_filename, 'rt') as training_data_text:
        for line in training_data_text:
            parts = line.strip().split('\t')
            label = parts[0]
            signal = parts[1]
            if random.random() < test_fraction:
                testing_data.append((label, signal))
            else:
                training_data.append((label, signal))
    print('done')
    print(' ', len(training_data), 'training samples')
    print(' ', len(testing_data), 'testing samples')

    print()
    print('Preparing signal data', end='')
    training_signals, training_labels = load_data_into_numpy(training_data)
    testing_signals, testing_labels = load_data_into_numpy(testing_data)
    print(' done')
    print(' ', len(training_signals), 'training samples')
    print(' ', len(testing_signals), 'testing samples')
    print('\n')

    return training_signals, training_labels, testing_signals, testing_labels


def load_data_into_numpy(data_list):
    signals = np.empty([len(data_list), signal_length], dtype=float)
    labels = np.empty([len(data_list), barcode_count + 1], dtype=float)

    for i, data in enumerate(data_list):
        label, signal = data
        label = int(label)

        signal = [float(x) for x in signal.split(',')]
        good_signal, trim_pos = get_good_part_of_signal(signal, label)

        # # Plot the resulting signal (for debugging)
        # print(np.std(signal))
        # plt.plot(signal)
        # plt.axvline(x=trim_pos, color='red')
        # plt.axvline(x=trim_pos+signal_length, color='red')
        # plt.show()

        if len(good_signal) == 0:
            continue

        label_list = [0.0] * (barcode_count + 1)
        label_list[label] = 1.0

        signals[i] = good_signal
        labels[i] = label_list

        if i % 100 == 0:
            print('.', end='', flush=True)

    return signals, labels


def get_good_part_of_signal(signal, label):
    # If the label is 0, then this isn't a read start/end, but rather signal taken from
    # the middle of a read, so we just grab the middle of the available signal.
    if label == 0:
        trim_pos = (len(signal) // 2) - (signal_length // 2)

    # If the label is a barcode, then we want to start the signal right after the open
    # pore signal ends.
    else:
        trim_pos = find_signal_start_pos(signal)
        if trim_pos == 0:
            return np.empty(0), 0

    signal = signal[trim_pos:trim_pos + signal_length]
    if len(signal) < signal_length:
        return np.empty(0), 0

    # Normalise the signal to zero mean and unit stdev.
    mean = np.mean(signal)
    stdev = np.std(signal)
    signal = (signal - mean) / stdev

    return signal, trim_pos


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


if __name__ == '__main__':
    main()
