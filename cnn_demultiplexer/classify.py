
import sys
import pathlib
import h5py
import numpy as np
from keras.models import load_model
from .load_fast5s import find_all_fast5s, get_read_id_and_signal
from .trim_signal import normalise


def classify(args):
    model, signal_size = load_trained_model(args.model)

    input_type = determine_input_type(args.input)
    if input_type == 'directory':
        classify_fast5_files(find_fast5s_in_dir(args.input), model, signal_size, args)
    elif input_type == 'single_fast5':
        classify_fast5_files([args.input], model, signal_size, args)
    elif input_type == 'training_data':
        classify_training_data(args.input, model, signal_size, args)
    else:
        assert False


def load_trained_model(model_file):
    print('', file=sys.stderr)
    if not pathlib.Path(model_file).is_file():
        sys.exit('Error: {} does not exist'.format(model_file))
    print('Loading neural network... ', file=sys.stderr, end='', flush=True)
    model = load_model(model_file)
    print('done', file=sys.stderr)
    try:
        assert len(model.inputs) == 1
        input_shape = model.inputs[0].shape
        signal_size = int(input_shape[1])
        assert signal_size > 10
        assert input_shape[2] == 1
    except (AssertionError, IndexError):
        sys.exit('Error: model input has incorrect shape - are you sure that {} is a valid '
                 'model file?'.format(model_file))
    return model, signal_size


def classify_fast5_files(fast5_files, model, signal_size, args):
    print('', file=sys.stderr)

    # TO DO: progress bar

    print('\t'.join(['Read_ID',
                     'none', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                     'Barcode_call']))

    for fast5_batch in chunker(fast5_files, args.batch_size):
        read_ids = []
        start_read_signals = np.empty([len(fast5_batch), signal_size], dtype=float)
        end_read_signals = np.empty([len(fast5_batch), signal_size], dtype=float)

        for i, fast5_file in enumerate(fast5_batch):
            read_id, signal = get_read_id_and_signal(fast5_file)
            read_ids.append(read_id)

            start_signal = normalise(signal[:signal_size])
            end_signal = normalise(signal[-signal_size:])

            start_read_signals[i] = start_signal
            end_read_signals[i] = end_signal
        start_read_signals = np.expand_dims(start_read_signals, axis=2)
        labels = model.predict(start_read_signals, batch_size=args.batch_size)

        for i, read_id in enumerate(read_ids):
            barcode_probabilities = list(labels[i])
            assert(len(barcode_probabilities) == 13)

            max_prob = max(barcode_probabilities)
            if max_prob < args.min_barcode_score:
                barcode_call = 'none'
            elif barcode_probabilities.count(max_prob) != 1:  # Tie for best
                barcode_call = 'none'
            else:
                i = barcode_probabilities.index(max_prob)
                if i == 0:
                    barcode_call = 'none'
                else:
                    barcode_call = str(i)

            print('\t'.join([read_id] +
                            ['%.2f' % x for x in barcode_probabilities] +
                            [barcode_call]))


def classify_training_data(input_file, model, signal_size, args):
    print('', file=sys.stderr)
    quit()

    # labels = model.predict(signals, batch_size=batch_size)


def determine_input_type(input_file_or_dir):
    path = pathlib.Path(input_file_or_dir)
    if path.is_dir():
        return 'directory'
    if not path.is_file():
        sys.exit('Error: {} is neither a file nor a directory'.format(input_file_or_dir))
    try:
        f = h5py.File(input_file_or_dir, 'r')
        f.close()
        return 'single_fast5'
    except OSError:
        pass
    with open(input_file_or_dir) as f:
        first_line = f.readline()
    try:
        parts = first_line.split('\t')
        _ = int(parts[0])
        signals = [int(x) for x in parts[1].split(',')]
        assert len(signals) > 10
        return 'training_data'
    except (AssertionError, ValueError, IndexError):
        sys.exit('Error: could not determine input type')


def find_fast5s_in_dir(input_dir):
    print('', file=sys.stderr)
    print('Looking for fast5 files in {}... '.format(input_dir), file=sys.stderr, end='',
          flush=True)
    fast5_files = find_all_fast5s(input_dir)
    print(' done', file=sys.stderr)
    print('{} fast5s found'.format(len(fast5_files)), file=sys.stderr)
    return fast5_files


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
