
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='CNN demultiplexer for Oxford Nanopore reads')
    subparsers = parser.add_subparsers(dest='subparser_name')
    porechop_subparser(subparsers)
    balance_subparser(subparsers)
    train_subparser(subparsers)
    classify_subparser(subparsers)

    args = parser.parse_args()
    if args.subparser_name == 'porechop':
        from .porechop import training_data_from_porechop
        training_data_from_porechop(args)

    if args.subparser_name == 'balance':
        from .balance import balance_training_samples
        balance_training_samples(args)

    if args.subparser_name == 'train':
        from .train_network import train
        train(args)

    if args.subparser_name == 'classify':
        check_classify_arguments(args)
        from .classify import classify
        classify(args)


def porechop_subparser(subparsers):
    group = subparsers.add_parser('porechop', description='Prepare training data using Porechop')

    # Positional arguments
    group.add_argument('porechop_out', type=str,
                       help='A file containing the output of Porechop (must have been run with '
                            '--verbosity 3)')
    group.add_argument('fast5_dir', type=str,
                       help='The directory containing the fast5 files (will be searched '
                            'recursively, so can contain subdirectories)')

    # Optional arguments
    group.add_argument('--signal_size', type=int, required=False, default=1000,
                       help='Amount of signal (number of samples) that will be used in the CNN')
    group.add_argument('--max_start_end_margin', type=float, required=False, default=6000,
                       help="Up to this much of a read's start/end signal will be saved")
    group.add_argument('--min_signal_length', type=float, required=False, default=20000,
                       help='Reads with fewer than this many signals are excluded from the '
                            'training data')


def balance_subparser(subparsers):
    group = subparsers.add_parser('balance', description='Select balanced set of training samples')

    # Positional arguments
    group.add_argument('training_data', type=str,
                       help='Raw training data produced by the porechop command')
    group.add_argument('out_prefix', type=str,
                       help='Prefix for the output files (*_read_starts and *_read_ends)')

    # Optional arguments
    group.add_argument('--signal_size', type=int, required=False, default=1000,
                       help='Amount of signal (number of samples) that will be used in the CNN')
    group.add_argument('--none_bin_rate', type=float, required=False, default=0.333333,
                       help='This fraction of the training samples will be no barcode signal')
    group.add_argument('--plot', action='store_true',
                       help='Display the signal plots for each read (for debugging use)')


def train_subparser(subparsers):
    group = subparsers.add_parser('train', description='Train the CNN')

    # Positional arguments
    group.add_argument('training_data', type=str,
                       help='Balanced training data produced by the select command')
    group.add_argument('model_out', type=str,
                       help='Filename for the trained model')

    # Optional arguments
    group.add_argument('--signal_size', type=int, required=False, default=1000,
                       help='Amount of signal (number of samples) that will be used in the CNN')
    group.add_argument('--barcode_count', type=int, required=False, default=12,
                       help='The number of discrete barcodes')
    group.add_argument('--epochs', type=int, required=False, default=100,
                       help='Number of training epochs')
    group.add_argument('--batch_size', type=int, required=False, default=128,
                       help='Training batch size')
    group.add_argument('--test_fraction', type=float, required=False, default=0.1,
                       help='This fraction of the training samples will be used as a test set')


def classify_subparser(subparsers):
    group = subparsers.add_parser('classify', description='Classify reads using the CNN')

    # Positional arguments
    group.add_argument('input', type=str,
                       help='One of the following: a single fast5 file, a directory of fast5 '
                            'files (will be searched recursively) or a tab-delimited file of '
                            'training data')
    group.add_argument('model', type=str, nargs='+',
                       help='One or two model files produced by the train command')

    # Optional arguments
    group.add_argument('--fastq_file', type=str, required=False,
                       help='A fastq file (can be gzipped) of basecalled reads')
    group.add_argument('--fastq_dir', type=str, required=False,
                       help='A directory of fastq files (will be searched recursively, files can '
                            'be gzipped) of basecalled reads')
    group.add_argument('--fastq_out_dir', type=str, required=False,
                       help='Output directory for binned reads (must be used with either '
                            '--fastq_file or --fastq_dir')
    group.add_argument('--batch_size', type=int, required=False, default=128,
                       help='CNN batch size')
    group.add_argument('--scan_size', type=float, required=False, default=6000,
                       help="This much of a read's start/end signal will examined for barcode "
                            "signals")
    group.add_argument('--score_diff', type=float, required=False, default=0.5,
                       help='For a read to be classified, there must be this much difference '
                            'between the best and second-best barcode scores')
    group.add_argument('--require_both', action='store_true',
                       help='When classifying reads using two models (read start and read end) '
                            'require both barcode calls to match to make the final call')
    group.add_argument('--verbose', action='store_true',
                       help='Include the CNN probabilities for all barcodes in the results '
                            '(default: just show the final barcode call)')


def check_classify_arguments(args):
    if args.fastq_file is not None and args.fastq_dir is not None:
        sys.exit('Error: --fastq_file and --fastq_dir are mutually exclusive')
    if args.fastq_file is not None and args.fastq_out_dir is None:
        sys.exit('Error: --fastq_out_dir must be used with --fastq_file')
    if args.fastq_dir is not None and args.fastq_out_dir is None:
        sys.exit('Error: --fastq_out_dir must be used with --fastq_dir')
    if len(args.model) > 2:
        sys.exit('Error: you must provide exactly one or two trained model files')
    if args.score_diff <= 0.0 or args.score_diff > 1.0:
        sys.exit('Error: --score_diff must be in the range (0, 1] (greater than 0 and less than or '
                 'equal to 1)')


if __name__ == '__main__':
    main()
