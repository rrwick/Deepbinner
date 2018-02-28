

import re
import sys
from .load_fast5s import get_read_id_and_signal, find_all_fast5s
from .trim_signal import too_much_open_pore


def training_data_from_porechop(args):
    barcode_calls = get_porechop_barcode_calls(args.porechop_out)

    print('\t'.join(['Read_ID', 'Barcode_bin',
                     'Start_read_signal', 'Middle_read_signal', 'End_read_signal']))

    fast5_files = find_all_fast5s(args.fast5_dir)

    print('', file=sys.stderr)
    print('Loading signal from {} fast5 files'.format(len(fast5_files)), end='', file=sys.stderr)

    i, short_count, bad_signal_count = 0, 0, 0

    for fast5_file in fast5_files:
        i += 1
        if i % 100 == 0:
            print('.', end='', file=sys.stderr, flush=True)

        read_id, signal = get_read_id_and_signal(fast5_file)

        if read_id is None or read_id not in barcode_calls:
            continue
        if len(signal) < args.min_signal_length:
            short_count += 1
            continue

        barcode_bin = barcode_calls[read_id]
        bad_signal = False

        # The middle signal is simply the centre-most signal in the read.
        middle_pos = len(signal) // 2
        middle_1 = middle_pos - (args.signal_size // 2)
        middle_2 = middle_pos + (args.signal_size // 2)
        middle_signal = signal[middle_1:middle_2]

        start_margin = args.signal_size * 2
        while too_much_open_pore(signal[:start_margin]):
            start_margin += args.signal_size
            if start_margin > args.max_start_end_margin:
                bad_signal = True
                break

        end_margin = args.signal_size * 2
        while too_much_open_pore(signal[-end_margin:]):
            end_margin += args.signal_size
            if end_margin > args.max_start_end_margin:
                bad_signal = True
                break

        if bad_signal:
            bad_signal_count += 1
            continue

        start_signal = signal[:start_margin]
        end_signal = signal[-end_margin:]

        start_signal_str = ','.join(str(x) for x in start_signal)
        middle_signal_str = ','.join(str(x) for x in middle_signal)
        end_signal_str = ','.join(str(x) for x in end_signal)

        print('\t'.join([read_id, str(barcode_bin),
                         start_signal_str, middle_signal_str, end_signal_str]))

    print(' done', file=sys.stderr)
    print('skipped ' + str(short_count) + ' reads for being too short\n', file=sys.stderr)
    print('skipped ' + str(bad_signal_count) + ' reads for bad signal stdev\n', file=sys.stderr)


def get_porechop_barcode_calls(porechop_output_filename):
    print('', file=sys.stderr)
    print('Getting barcode calls from Porechop output... ', file=sys.stderr, end='', flush=True)
    barcode_calls = {}
    with open(porechop_output_filename, 'rt') as porechop_output:
        for line in porechop_output:
            m = re.search(r'^\w{8}-\w{4}-\w{4}-\w{4}-\w{12}', line)
            if m:
                read_id = m.group()

                # Gather up the read's info in a list of strings.
                read_info = []
                while True:
                    try:
                        line = next(porechop_output).strip()
                    except StopIteration:
                        break
                    if line:
                        read_info.append(line)
                    else:  # empty line indicates the end of one read's info
                        break
                barcode_bin = get_final_barcode_call(read_info)
                if barcode_bin is not None:
                    barcode_calls[read_id] = barcode_bin
    print(' done', file=sys.stderr)
    return barcode_calls


def get_final_barcode_call(read_info):
    assert read_info[-1].startswith('final barcode call:')
    barcode_bin = read_info[-1].split('final barcode call:')[-1].strip()
    assert barcode_bin.startswith('BC') or barcode_bin == 'none'
    if barcode_bin.startswith('BC'):
        return str(int(barcode_bin[2:]))
    else:
        return None
