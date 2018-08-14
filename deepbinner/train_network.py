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
import numpy as np
import sys

from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

from .network_architecture import build_network
from .trim_signal import normalise
from .classify import load_trained_model


def train(args):
    print()
    class_count = determine_class_count(args.train)
    signal_size = determine_signal_size(args.train)
    train_steps, val_steps = get_steps_count(args.train, args.val, args.batch_size)

    # If the user provided a model to start with, we load that.
    if args.model_in:
        model, input_size, output_size = load_trained_model(args.model_in, out_dest=sys.stdout)
        if input_size != signal_size:
            sys.exit('Error: the provided model\'s input size ({}) is not equal to the '
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

    model.compile(optimizer='nadam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    training_data = data_generator(args.train, signal_size, args.batch_size, class_count,
                                   augmentation=args.aug)
    validation_data = data_generator(args.val, signal_size, args.batch_size, class_count,
                                     augmentation=1.0)

    checkpoint = ModelCheckpoint(args.model_out, monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')

    model.fit_generator(training_data, steps_per_epoch=min(train_steps, args.batches_per_epoch),
                        epochs=args.epochs, validation_data=validation_data,
                        validation_steps=val_steps, callbacks=[checkpoint])


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


def get_steps_count(training_filename, validation_filename, batch_size):
    with open(training_filename, 'rt') as training_file:
        training_count = sum(1 for _ in training_file)
    with open(validation_filename, 'rt') as validation_file:
        validation_count = sum(1 for _ in validation_file)
    if training_count < batch_size:
        sys.exit('Error: the number of training samples is smaller than the batch size')
    if validation_count < batch_size:
        sys.exit('Error: the number of validation samples is smaller than the batch size')
    print('Data summary:')
    print('    training samples: {}'.format(training_count))
    print('  validation samples: {}'.format(validation_count))
    print('          batch_size: {}'.format(batch_size))
    print()
    return training_count // batch_size, validation_count // batch_size


def data_generator(data_filename, signal_size, batch_size, class_count, augmentation=1.0):
    """
    This generator indefinitely yields batches of signals and labels. It enables the use of Keras's
    fit_generator function which gives a couple benefits:
    * Can train arbitrarily large epochs because not all data needs to be held in memory.
    * Can do data augmentation on the CPU for the next batch while the current batch is training
      on the GPU (more efficient).
    """
    augmentation_chance = 1.0 - (1.0 / augmentation)

    data = []
    with open(data_filename, 'rt') as data_file:
        for line in data_file:
            parts = line.strip().split('\t')
            label = parts[0]
            signal = np.array([int(x) for x in parts[1].split(',')])
            assert len(signal) == signal_size
            data.append((label, normalise(signal)))
    random.shuffle(data)

    current_sample_index = 0
    while True:
        batch_signals = np.empty([batch_size, signal_size, 1], dtype=float)
        batch_labels = np.empty([batch_size, class_count], dtype=float)
        for i in range(batch_size):
            try:
                label, signal = data[current_sample_index]
            except IndexError:
                random.shuffle(data)
                current_sample_index = 0
                label, signal = data[0]
            current_sample_index += 1

            if random.random() < augmentation_chance:
                signal = modify_signal(signal)

            label = to_categorical(label, num_classes=class_count)
            batch_labels[i] = label
            batch_signals[i] = np.expand_dims(signal, axis=2)

        yield batch_signals, batch_labels


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
