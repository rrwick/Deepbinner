
import sys


def refine_training_samples(args):
    with open(args.training_data) as old_training_data, open(args.classification_data) as classes:
        _ = next(classes)  # Skip the header line

        match_count, total_count = 0, 0
        for training_line, class_line in zip(old_training_data, classes):
            classes_parts = class_line.rstrip().split('\t')
            train_barcode = training_line.split('\t')[0]
            class_barcode = classes_parts[1]

            # Sanity check to make sure the two files' lines are aligned.
            assert classes_parts[0].split('_')[-1] == train_barcode

            if train_barcode == '0':
                train_barcode = 'none'

            if train_barcode == class_barcode:
                print(training_line, end='')
                match_count += 1
            total_count += 1

            print('\rMatches: {} / {} ({:.1f}%)'.format(match_count, total_count,
                                                        100.0 * match_count / total_count),
                  file=sys.stderr, end='')

        print('', file=sys.stderr)
