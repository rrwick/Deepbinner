
import random
import time
import numpy as np

from keras.layers import Input
from keras.models import Model
from .network_architecture import random_080_norm_and_drop_after_bottleneck
from .trim_signal import normalise


def train(args):
    print()
    class_count = args.barcode_count + 1

    inputs = Input(shape=(args.signal_size, 1))
    predictions = random_080_norm_and_drop_after_bottleneck(inputs, class_count)

    model = Model(inputs=inputs, outputs=predictions)
    model.summary()
    print('\n')

    signals, labels = load_training_set(args.training_data, args.signal_size, class_count)

    # Partition off 10% of the data for use as a validation set.
    validation_count = int(len(signals) * args.test_fraction)
    if validation_count > 0:
        validation_signals = signals[:validation_count]
        validation_labels = labels[:validation_count]
        training_signals = signals[validation_count:]
        training_labels = labels[validation_count:]
    else:
        validation_signals, validation_labels = [], []
        training_signals, training_labels = signals, labels

    print('Training/validation split: {}, {}'.format(len(training_signals),
                                                     len(validation_signals)))

    training_signals, training_labels = augment_data(training_signals, training_labels,
                                                     args.signal_size, class_count,
                                                     augmentation_factor=3)

    training_signals = np.expand_dims(training_signals, axis=2)
    validation_signals = np.expand_dims(validation_signals, axis=2)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    before_time = time.time()
    hist = model.fit(training_signals, training_labels,
                     epochs=args.epochs,
                     batch_size=args.batch_size,
                     shuffle=True,
                     validation_data=(validation_signals, validation_labels))
    after_time = time.time()
    training_time_minutes = (after_time - before_time) / 60

    final_acc = hist.history['acc'][-1]
    final_val_acc = hist.history['val_acc'][-1]
    mean_5_best_acc = np.mean(sorted(hist.history['acc'][-5:]))
    mean_5_best_val_acc = np.mean(sorted(hist.history['val_acc'][-5:]))
    prediction_time_ms = time_model_prediction(model, validation_signals)

    print('\n')
    print('\t'.join(['final_acc',
                     'final_val_acc',
                     'mean_5_best_acc',
                     'mean_5_best_val_acc',
                     'training_time_minutes',
                     'prediction_time_ms']))
    print('\t'.join(['%.4f' % final_acc,
                     '%.4f' % final_val_acc,
                     '%.4f' % mean_5_best_acc,
                     '%.4f' % mean_5_best_val_acc,
                     '%.4f' % training_time_minutes,
                     '%.4f' % prediction_time_ms]))

    model.save(args.model_out)
    print()


def load_training_set(training_data_filename, signal_size, class_count):
    training_data = []

    print()
    print('Loading data from file... ', end='')
    with open(training_data_filename, 'rt') as training_data_text:
        for line in training_data_text:
            parts = line.strip().split('\t')
            training_data.append((parts[0], parts[1]))
    print('done')
    print(' ', len(training_data), 'samples')

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
        assert len(signal) == signal_size

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
    print('Final training data:', len(augmented_signals), 'samples')
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
