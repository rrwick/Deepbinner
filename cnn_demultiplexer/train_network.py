
import random

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten
from keras.models import Sequential
from .trim_signal import find_signal_start_pos


# Some hard-coded parameters
optimizer = 'rmsprop'
loss = 'binary_crossentropy'
activation = 'relu'


def train(args):
    class_count = args.barcode_count + 1

    signals, labels = load_training_set(args.training_data, args.signal_size, class_count)

    model = Sequential()

    model.add(Conv1D(filters=16, kernel_size=3, activation=activation, padding='same',
                     input_shape=(args.signal_size, 1)))
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
    model.add(Dense(class_count, activation='softmax'))



    for layer in model.layers:
        print()
        print(layer.input)
        print(layer.output)
        print(layer.get_config())
        print()

    quit()


    model.compile(optimizer=optimizer, loss=loss)

    try:
        model.fit(signals, labels,
                  epochs=args.epochs,
                  batch_size=256,
                  shuffle=True,
                  validation_split=args.test_fraction)
    except KeyboardInterrupt:
        pass

    quit()


def load_training_set(training_data_filename, signal_size, class_count):
    random.seed(0)

    training_data = []

    print()
    print('Loading data from file... ', end='')
    with open(training_data_filename, 'rt') as training_data_text:
        for line in training_data_text:
            parts = line.strip().split('\t')
            training_data.append((parts[0], parts[1]))
    print('done')
    print(' ', len(training_data), 'training samples')

    random.shuffle(training_data)

    print()
    print('Preparing signal data', end='')
    signals, labels = load_data_into_numpy(training_data, signal_size, class_count)
    print(' done')
    print()

    return signals, labels


def load_data_into_numpy(data_list, signal_size, class_count):
    signals = np.empty([len(data_list), signal_size], dtype=float)
    labels = np.empty([len(data_list), class_count], dtype=float)

    for i, data in enumerate(data_list):
        label, signal = data
        label = int(label)

        signal = [float(x) for x in signal.split(',')]
        good_signal, trim_pos = get_good_part_of_signal(signal, label, signal_size)

        # # Plot the resulting signal (for debugging)
        # print(np.std(signal))
        # plt.plot(signal)
        # plt.axvline(x=trim_pos, color='red')
        # plt.axvline(x=trim_pos+signal_size, color='red')
        # plt.show()

        if len(good_signal) == 0:
            continue

        label_list = [0.0] * class_count
        label_list[label] = 1.0

        signals[i] = good_signal
        labels[i] = label_list

        if i % 100 == 0:
            print('.', end='', flush=True)

    return signals, labels


def get_good_part_of_signal(signal, label, signal_size):
    # If the label is 0, then this isn't a read start/end, but rather signal taken from
    # the middle of a read, so we just grab the middle of the available signal.
    if label == 0:
        trim_pos = (len(signal) // 2) - (signal_size // 2)

    # If the label is a barcode, then we want to start the signal right after the open
    # pore signal ends.
    else:
        trim_pos = find_signal_start_pos(signal)
        if trim_pos == 0:
            return np.empty(0), 0

    signal = signal[trim_pos:trim_pos + signal_size]
    if len(signal) < signal_size:
        return np.empty(0), 0

    # Normalise the signal to zero mean and unit stdev.
    mean = np.mean(signal)
    stdev = np.std(signal)
    signal = (signal - mean) / stdev

    return signal, trim_pos
