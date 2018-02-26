
import argparse


def main():
    parser = argparse.ArgumentParser(description='CNN demultiplexer for Oxford Nanopore reads')
    subparsers = parser.add_subparsers(dest='subparser_name')
    porechop_subparser(subparsers)
    balance_subparser(subparsers)
    train_subparser(subparsers)
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
    group.add_argument('out_prefix', type=str,
                       help='Prefix for the output files (*_model and *_loss)')

    # Optional arguments
    group.add_argument('--signal_size', type=int, required=False, default=1000,
                       help='Amount of signal (number of samples) that will be used in the CNN')
    group.add_argument('--barcode_count', type=int, required=False, default=12,
                       help='The number of discrete barcodes')
    group.add_argument('--epochs', type=int, required=False, default=500,
                       help='Number of training epochs')
    group.add_argument('--batch_size', type=int, required=False, default=128,
                       help='Training batch size')
    group.add_argument('--test_fraction', type=float, required=False, default=0.1,
                       help='This fraction of the training samples will be used as a test set')


if __name__ == '__main__':
    main()
