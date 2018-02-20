
import random
import time
import numpy as np

from keras.layers import Input, Dense, Dropout, Flatten
from keras.models import Model
from .network_architecture import classic_cnn, classic_cnn_with_bottlenecks, inception_network, \
    build_random_network


def train(args):
    print()
    class_count = args.barcode_count + 1

    inputs = Input(shape=(args.signal_size, 1))
    predictions = classic_cnn_with_bottlenecks(inputs, class_count)

    model = Model(inputs=inputs, outputs=predictions)
    model.summary()
    print('\n')

    signals, labels = load_training_set(args.training_data, args.signal_size, class_count)

    # Partition off 10% of the data for use as a validation set.
    validation_count = len(signals) // 10
    validation_signals = signals[:validation_count]
    validation_labels = labels[:validation_count]
    training_signals = signals[validation_count:]
    training_labels = labels[validation_count:]

    print('Training/validation split: {}, {}'.format(len(training_signals),
                                                     len(validation_signals)))

    training_signals, training_labels = augment_data(training_signals, training_labels,
                                                     args.signal_size, class_count,
                                                     augmentation_factor=3)

    training_signals = np.expand_dims(training_signals, axis=2)
    validation_signals = np.expand_dims(validation_signals, axis=2)

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    before_time = time.time()
    hist = model.fit(training_signals, training_labels,
                     epochs=args.epochs,
                     batch_size=args.batch_size,
                     shuffle=True,
                     validation_data=(validation_signals, validation_labels))
    after_time = time.time()
    elapsed_minutes = (after_time - before_time) / 60

    print('\n')
    print('Final validation loss:    ', '%.4f' % hist.history['val_loss'][-1])
    print('Final validation accuracy:', '%.4f' % hist.history['val_acc'][-1])
    print('Training time (minutes):  ', '%.2f' % elapsed_minutes)

    time_model_prediction(model, signals)

    model.save(args.out_prefix + '_model')
    save_history_to_file(args.out_prefix, hist.history)
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

        # Normalise to zero mean and unit stdev.
        mean = np.mean(signal)
        stdev = np.std(signal)
        signal = (signal - mean) / stdev

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
    print('Prediction time (ms/read):', '%.4f' % min_time)


def save_history_to_file(out_prefix, history):
    with open(out_prefix + '_loss', 'wt') as loss_file:
        loss_file.write('Epoch\tTraining_loss\tValidation_loss\t'
                        'Training_accuracy\tValidation_accuracy\n')
        for i, train_loss in enumerate(history['loss']):
            loss_file.write(str(i))
            loss_file.write('\t')
            loss_file.write(str(train_loss))
            loss_file.write('\t')
            loss_file.write(str(history['val_loss'][i]))
            loss_file.write('\t')
            loss_file.write(str(history['acc'][i]))
            loss_file.write('\t')
            loss_file.write(str(history['val_acc'][i]))
            loss_file.write('\n')


def augment_data(signals, labels, signal_size, class_count, augmentation_factor):
    print()
    if augmentation_factor == 1:
        print('Not performing data augmentation')
        return signals, labels

    print('Augmenting training data by a factor of', augmentation_factor, end='')
    data_count = len(signals)
    augmented_data_count = augmentation_factor * data_count
    augmented_signals = np.empty([augmented_data_count, signal_size], dtype=float)
    augmented_labels = np.empty([augmented_data_count, class_count], dtype=float)

    i = 0
    for signal, label in zip(signals, labels):
        augmented_signals[i] = signal
        augmented_labels[i] = label
        if i % 1000 == 0:
            print('.', end='', flush=True)
        i += 1
        for _ in range(augmentation_factor-1):
            augmented_signals[i] = modify_signal(signal)
            augmented_labels[i] = label
            i += 1

    assert i == augmented_data_count

    print()
    print('  final training data:', len(augmented_signals), 'samples')
    print()

    # Plot signals (for debugging)
    for signal in augmented_signals:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 5))
        fig.add_subplot(1, 1, 1)
        plt.plot(signal)
        plt.show()

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
