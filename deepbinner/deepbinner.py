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

import argparse
import sys
from .help_formatter import MyParser, MyHelpFormatter
from .version import __version__


def main():
    parser = MyParser(description='Deepbinner: a deep convolutional neural network '
                                  'barcode demultiplexer for Oxford Nanopore reads',
                      formatter_class=MyHelpFormatter, add_help=False)

    subparsers = parser.add_subparsers(title='Commands', dest='subparser_name')
    classify_subparser(subparsers)
    bin_subparser(subparsers)
    realtime_subparser(subparsers)
    porechop_subparser(subparsers)
    balance_subparser(subparsers)
    train_subparser(subparsers)
    refine_subparser(subparsers)

    longest_choice_name = max(len(c) for c in subparsers.choices)
    subparsers.help = 'R|'
    for choice, choice_parser in subparsers.choices.items():
        padding = ' ' * (longest_choice_name - len(choice))
        subparsers.help += choice + ': ' + padding
        d = choice_parser.description
        subparsers.help += d[0].lower() + d[1:]  # don't capitalise the first letter
        subparsers.help += '\n'

    help_args = parser.add_argument_group('Help')
    help_args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                           help='Show this help message and exit')
    help_args.add_argument('--version', action='version', version=__version__,
                           help="Show program's version number and exit")

    # If no arguments were used, print the base-level help which lists possible commands.
    if len(sys.argv) == 1:
        parser.print_help(file=sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.subparser_name == 'classify':
        check_classify_and_realtime_arguments(args)
        from .classify import classify
        classify(args)

    if args.subparser_name == 'bin':
        from .bin import bin_reads
        bin_reads(args)

    if args.subparser_name == 'realtime':
        check_classify_and_realtime_arguments(args)
        from .realtime import realtime
        realtime(args)

    elif args.subparser_name == 'porechop':
        from .porechop import training_data_from_porechop
        training_data_from_porechop(args)

    elif args.subparser_name == 'balance':
        check_balance_arguments(args)
        from .balance import balance_training_samples
        balance_training_samples(args)

    elif args.subparser_name == 'train':
        from .train_network import train
        train(args)

    elif args.subparser_name == 'refine':
        from .refine import refine_training_samples
        refine_training_samples(args)


def classify_subparser(subparsers):
    group = subparsers.add_parser('classify', description='Classify fast5 reads',
                                  formatter_class=MyHelpFormatter, add_help=False)

    positional_args = group.add_argument_group('Positional')
    positional_args.add_argument('input', type=str,
                                 help='One of the following: a single fast5 file, a directory of '
                                      'fast5 files (will be searched recursively) or a '
                                      'tab-delimited file of training data')
    classify_and_realtime_options(group)

    other_args = group.add_argument_group('Other')
    other_args.add_argument('--verbose', action='store_true',
                            help='Include the output probabilities for all barcodes in the '
                                 'results (default: just show the final barcode call)')
    other_args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                            help='Show this help message and exit')


def classify_and_realtime_options(group):
    """
    A few options are used in both the classify and realtime command, so they are described in this
    separate function.
    """
    model_args = group.add_argument_group('Models (at least one is required)')
    model_args.add_argument('-s', '--start_model', type=str, required=False,
                            help='Model trained on the starts of reads')
    model_args.add_argument('-e', '--end_model', type=str, required=False,
                            help='Model trained on the ends of reads')

    barcode_args = group.add_argument_group('Barcoding')
    barcode_args.add_argument('--scan_size', type=float, required=False, default=6144,
                              help="This much of a read's start/end signal will examined for "
                                   "barcode signals")
    barcode_args.add_argument('--score_diff', type=float, required=False, default=0.5,
                              help='For a read to be classified, there must be this much '
                                   'difference between the best and second-best barcode scores')
    barcode_args.add_argument('--require_both', action='store_true',
                              help='When classifying reads using two models (read start and read '
                                   'end) require both barcode calls to match to make the final '
                                   'call (default: a call on either the read start or read end is '
                                   'sufficient)')

    perf_args = group.add_argument_group('Performance')
    perf_args.add_argument('--batch_size', type=int, required=False, default=256,
                           help='Neural network batch size')
    perf_args.add_argument('--intra_op_parallelism_threads', type=int, required=False, default=12,
                           help='TensorFlow\'s intra_op_parallelism_threads config option')
    perf_args.add_argument('--inter_op_parallelism_threads', type=int, required=False, default=1,
                           help='TensorFlow\'s inter_op_parallelism_threads config option')
    perf_args.add_argument('--device_count', type=int, required=False, default=1,
                           help='TensorFlow\'s device_count config option')
    perf_args.add_argument('--omp_num_threads', type=int, required=False, default=12,
                           help='OMP_NUM_THREADS environment variable value')


def bin_subparser(subparsers):
    group = subparsers.add_parser('bin', description='Bin fasta/q reads',
                                  formatter_class=MyHelpFormatter, add_help=False)

    required_args = group.add_argument_group('Required')
    required_args.add_argument('--classes', type=str, required=True,
                               help='Deepbinner classification file (made with the deepbinner '
                                    'classify command)')
    required_args.add_argument('--reads', type=str, required=True,
                               help='FASTA or FASTQ reads')
    required_args.add_argument('--out_dir', type=str, required=True,
                               help='Directory to output binned read files')

    other_args = group.add_argument_group('Other')
    other_args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                            help='Show this help message and exit')


def realtime_subparser(subparsers):
    group = subparsers.add_parser('realtime', description='Sort fast5 files during sequencing',
                                  formatter_class=MyHelpFormatter, add_help=False)

    required_args = group.add_argument_group('Required')
    required_args.add_argument('--in_dir', type=str, required=True,
                               help='Directory where sequencer deposits fast5 files')
    required_args.add_argument('--out_dir', type=str, required=True,
                               help='Directory to output binned fast5 files')

    classify_and_realtime_options(group)

    other_args = group.add_argument_group('Other')
    other_args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                            help='Show this help message and exit')


def porechop_subparser(subparsers):
    group = subparsers.add_parser('porechop', description='Prepare training data using Porechop',
                                  formatter_class=MyHelpFormatter)

    # Positional arguments
    group.add_argument('porechop_out', type=str,
                       help='A file containing the output of Porechop (must have been run with '
                            '--verbosity 3)')
    group.add_argument('fast5_dir', type=str,
                       help='The directory containing the fast5 files (will be searched '
                            'recursively, so can contain subdirectories)')

    # Optional arguments
    group.add_argument('--signal_size', type=int, required=False, default=1024,
                       help='Amount of signal (number of samples) that will be used in the neural '
                            'network')
    group.add_argument('--max_start_end_margin', type=float, required=False, default=6000,
                       help="Up to this much of a read's start/end signal will be saved")
    group.add_argument('--min_signal_length', type=float, required=False, default=20000,
                       help='Reads with fewer than this many signals are excluded from the '
                            'training data')


def balance_subparser(subparsers):
    group = subparsers.add_parser('balance', description='Select balanced training set',
                                  formatter_class=MyHelpFormatter)

    # Positional arguments
    group.add_argument('training_data', type=str,
                       help='Raw training data produced by the porechop command')
    group.add_argument('out_prefix', type=str,
                       help='Prefix for the output files (*_read_starts and *_read_ends)')

    # Optional arguments
    group.add_argument('--barcodes', type=str,
                       help='A comma-delimited list of which barcodes to include (default: '
                            'include all barcodes)')
    group.add_argument('--signal_size', type=int, required=False, default=1024,
                       help='Amount of signal (number of samples) that will be used in the neural '
                            'network')
    group.add_argument('--none_bin_rate', type=float, required=False, default=0.333333,
                       help='This fraction of the training samples will be no barcode signal')
    group.add_argument('--plot', action='store_true',
                       help='Display the signal plots for each read (for debugging use)')


def train_subparser(subparsers):
    group = subparsers.add_parser('train', description='Train the neural network',
                                  formatter_class=MyHelpFormatter)

    # Positional arguments
    group.add_argument('training_data', type=str,
                       help='Balanced training data produced by the balance command')
    group.add_argument('model_out', type=str,
                       help='Filename for the trained model')

    # Optional arguments
    group.add_argument('--signal_size', type=int, required=False, default=1024,
                       help='Amount of signal (number of samples) that will be used in the neural '
                            'network')
    group.add_argument('--epochs', type=int, required=False, default=100,
                       help='Number of training epochs')
    group.add_argument('--aug', type=int, required=False, default=2,
                       help='Data augmentation factor (1 = no augmentation)')
    group.add_argument('--batch_size', type=int, required=False, default=128,
                       help='Training batch size')
    group.add_argument('--test_fraction', type=float, required=False, default=0.1,
                       help='This fraction of the training samples will be used as a test set')


def refine_subparser(subparsers):
    group = subparsers.add_parser('refine', description='Refine the training set',
                                  formatter_class=MyHelpFormatter)

    # Positional arguments
    group.add_argument('training_data', type=str,
                       help='Balanced training data produced by the balance command')
    group.add_argument('classification_data', type=str,
                       help='Training data barcode calls produced by the classify command')


def check_classify_and_realtime_arguments(args):
    model_count = (0 if args.start_model is None else 1) + (0 if args.end_model is None else 1)
    if model_count == 0:
        sys.exit('Error: you must provide at least one model')
    if model_count < 2 and args.require_both:
        sys.exit('Error: --require_both can only be used with two models (start and end)')
    if args.score_diff <= 0.0 or args.score_diff > 1.0:
        sys.exit('Error: --score_diff must be in the range (0, 1] (greater than 0 and less than or '
                 'equal to 1)')


def check_balance_arguments(args):
    if args.barcodes is not None:
        args.barcodes = args.barcodes.split(',')
        try:
            _ = [int(x) for x in args.barcodes]
        except ValueError:
            sys.exit('Error: if used, --barcodes must be a comma-delimited list of numbers (no '
                     'spaces)')


if __name__ == '__main__':
    main()
