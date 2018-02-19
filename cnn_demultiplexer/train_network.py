
import random
import time
import numpy as np

from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten
from keras.models import Sequential


# Some hard-coded parameters
optimizer = 'rmsprop'
loss = 'binary_crossentropy'
activation = 'relu'


def train(args):
    class_count = args.barcode_count + 1

    signals, labels = load_training_set(args.training_data, args.signal_size, class_count)
    signals = np.expand_dims(signals, axis=2)

    print('Building network:')
    model = Sequential()

    model.add(Conv1D(filters=8, kernel_size=3, activation=activation, padding='same',
                     input_shape=(args.signal_size, 1)))
    model.add(Conv1D(filters=8, kernel_size=3, activation=activation))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    
    model.add(Conv1D(filters=16, kernel_size=3, activation=activation, padding='same'))
    model.add(Conv1D(filters=16, kernel_size=3, activation=activation))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    
    model.add(Conv1D(filters=16, kernel_size=3, activation=activation, padding='same'))
    model.add(Conv1D(filters=16, kernel_size=3, activation=activation))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(100, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(class_count, activation='softmax'))

    model.summary()
    print('\n')

    model.compile(optimizer=optimizer, loss=loss)

    before_time = time.time()
    hist = model.fit(signals, labels,
                     epochs=args.epochs,
                     batch_size=args.batch_size,
                     shuffle=True,
                     validation_split=args.test_fraction)
    after_time = time.time()
    elapsed_minutes = (after_time - before_time) / 60

    print('\n')
    print('Final validation loss:    ', '%.4f' % hist.history['val_loss'][-1])
    print('Training time (minutes):  ', '%.2f' % elapsed_minutes)

    time_model_prediction(model, signals)

    model.save(args.model_out)
    print()


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

        # Normalise to zero mean and unit stdev.
        mean = np.mean(signal)
        stdev = np.std(signal)
        signal = (signal - mean) / stdev

        label_list = [0.0] * class_count
        label_list[label] = 1.0

        signals[i] = signal
        labels[i] = label_list

        if i % 100 == 0:
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
