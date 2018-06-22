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

import random
import time
import numpy as np
import sys

from keras.layers import Input
from keras.models import Model
from .network_architecture import build_network
from .trim_signal import normalise
from .classify import load_trained_model


def train(args):
    print()
    class_count = determine_class_count(args.training_data)
    signal_size = determine_signal_size(args.training_data)

    # If the user provided a model to start with, we load that
    if args.model_in:
        model, input_size, output_size = load_trained_model(args.model_in, out_dest=sys.stdout)
        if input_size != signal_size:
            sys.exit('Error: the provided model\'s input size ({}) are not equal to the '
                     'training data\'s size ({})'.format(input_size, signal_size))
        if output_size != class_count:
            sys.exit('Error: the provided model\'s output classes ({}) are not equal to the '
                     'training data\'s classes ({})'.format(output_size, class_count))

    # If a starting model wasn't provided, a fresh one is built from scratch.
    else:
        inputs = Input(shape=(signal_size, 1))
        predictions = build_network(inputs, class_count)
        model = Model(inputs=inputs, outputs=predictions)
        model.summary()
        print()

    training_signals, training_labels, validation_signals, validation_labels = \
        load_training_and_validation_data(args, class_count, signal_size)

    validation_signals = np.expand_dims(validation_signals, axis=2)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    for _ in range(args.epochs):
        # Augmentation is redone after each epoch.
        augmented_signals, augmented_labels = augment_data(training_signals, training_labels,
                                                           signal_size, class_count,
                                                           augmentation_factor=args.aug)
        augmented_signals = np.expand_dims(augmented_signals, axis=2)

        model.fit(augmented_signals, augmented_labels, epochs=1, batch_size=args.batch_size,
                  shuffle=True, validation_data=(validation_signals, validation_labels))
        model.save(args.model_out)
    print()


def load_training_and_validation_data(args, class_count, signal_size):
    signals, labels = load_training_set(args.training_data, signal_size, class_count)
    validation_count = int(len(signals) * args.val_fraction)

    # If the user supplied separate validation data, then just use that.
    if args.val_data:
        validation_signals, validation_labels = load_training_set(args.val_data, signal_size,
                                                                  class_count, label='validation')
        return signals, labels, validation_signals, validation_labels

    # Partition off some of the data for use as a validation set.
    elif validation_count > 0:
        validation_signals = signals[:validation_count]
        validation_labels = labels[:validation_count]
        training_signals = signals[validation_count:]
        training_labels = labels[validation_count:]
        print('Training/validation split: {}, {}'.format(len(training_signals),
                                                         len(validation_signals)))
        return training_signals, training_labels, validation_signals, validation_labels

    # If we are training using all the data, then we don't divide into training and test. Instead,
    # the same data (all of it) is used for both.
    else:
        return signals, labels, signals, labels


def determine_class_count(training_data_filename):
    barcodes = set()
    with open(training_data_filename, 'rt') as training_data_text:
        for line in training_data_text:
            parts = line.strip().split('\t')
            barcodes.add(parts[0])
    try:
        barcodes = [int(x) for x in barcodes]
    except ValueError:
        sys.exit('Error: a non-integer barcode class was encountered '
                 'in {}'.format(training_data_filename))
    class_count = max(barcodes) + 1
    print('Number of possible barcode classifications = {}'.format(class_count))
    print('  (1-{} plus a no-barcode class)'.format(class_count-1))
    print()
    return class_count


def determine_signal_size(training_data_filename):
    signal_sizes = []
    i = 0
    with open(training_data_filename, 'rt') as training_data_text:
        for line in training_data_text:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                sys.exit('Error: training signal not formatted correctly')
            signal_size = len(parts[1].split(','))
            signal_sizes.append(signal_size)
            i += 1
            if i >= 100:  # don't bother looking at all of them
                break
    min_size, max_size = min(signal_sizes), max(signal_sizes)
    if min_size != max_size:
        sys.exit('Error: inconsistent signal sizes in training data')

    print('Training data signal length = {}'.format(max_size))
    print()
    return max_size


def load_training_set(training_data_filename, signal_size, class_count, label='training'):
    training_data = []

    print()
    print('Loading {} data from file... '.format(label), end='')
    with open(training_data_filename, 'rt') as training_data_text:
        for line in training_data_text:
            parts = line.strip().split('\t')
            training_data.append((parts[0], parts[1]))
    print('done')
    print(' ', len(training_data), 'samples')

    random.shuffle(training_data)

    print()
    print('Preparing {} signal data'.format(label), end='')
    signals, labels = load_data_into_numpy(training_data, signal_size, class_count)
    print(' done')

    return signals, labels


def load_data_into_numpy(data_list, signal_size, class_count):
    signals = np.empty([len(data_list), signal_size], dtype=float)
    labels = np.empty([len(data_list), class_count], dtype=float)

    for i, data in enumerate(data_list):
        label, signal = data
        label = int(label)

        signal = [float(x) for x in signal.split(',')]
        if len(signal) != signal_size:
            sys.exit('Error: signal length in training data is inconsistent (expected signal '
                     'of {} but got {}'.format(signal_size, len(signal)))

        signal = normalise(signal)

        label_list = [0.0] * class_count
        label_list[label] = 1.0

        signals[i] = signal
        labels[i] = label_list

        if i % 1000 == 0:
            print('.', end='', flush=True)

    return signals, labels


def time_model_prediction(model, signals):
    min_time = float('inf')
    for _ in range(10):
        before_time = time.time()
        model.predict(signals)
        after_time = time.time()
        elapsed_milliseconds = (after_time - before_time) * 1000
        milliseconds_per_read = elapsed_milliseconds / len(signals)
        min_time = min(min_time, milliseconds_per_read)
    return min_time


def augment_data(signals, labels, signal_size, class_count, augmentation_factor):
    print()
    if augmentation_factor <= 1:
        print('Not performing data augmentation')
        return signals, labels

    print('Augmenting training data by a factor of', augmentation_factor, end='')
    data_count = len(signals)
    augmented_data_count = augmentation_factor * data_count
    augmented_signals = np.empty([augmented_data_count, signal_size], dtype=float)
    augmented_labels = np.empty([augmented_data_count, class_count], dtype=float)

    i, j = 0, 0
    for signal, label in zip(signals, labels):
        augmented_signals[i] = signal
        augmented_labels[i] = label
        i += 1
        for _ in range(augmentation_factor-1):
            augmented_signals[i] = modify_signal(signal)
            augmented_labels[i] = label
            i += 1
        if j % 1000 == 0:
            print('.', end='', flush=True)
        j += 1

    assert i == augmented_data_count

    print('done')
    print()

    # Plot signals (for debugging)
    # for signal in augmented_signals:
    #     import matplotlib.pyplot as plt
    #     fig = plt.figure(figsize=(12, 5))
    #     fig.add_subplot(1, 1, 1)
    #     plt.plot(signal)
    #     plt.show()

    return augmented_signals, augmented_labels


def modify_signal(signal):
    modification_count = len(signal) * 0.5
    modification_count = int(round(modification_count / 2)) * 2
    modification_positions = random.sample(range(len(signal)), k=modification_count)
    half = int(modification_count / 2)
    duplication_positions = set(modification_positions[:half])
    deletion_positions = set(modification_positions[half:])

    new_signal = np.empty([len(signal)], dtype=float)
    j = 0
    for i, val in enumerate(signal):
        if i in duplication_positions:
            new_signal[j] = val
            j += 1
            new_signal[j] = val
            j += 1
        elif i in deletion_positions:
            pass
        else:
            new_signal[j] = val
            j += 1

    assert j == len(signal)
    return new_signal
