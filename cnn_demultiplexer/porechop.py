

import sys
import re
import os
import h5py
import numpy


def training_data_from_porechop(args):
    signals = get_signal_from_fast5s(args.fast5_dir, args.signal_size, args.stdev_threshold,
                                     args.max_start_end_margin, args.min_signal_length)

    print('\t'.join(['Read_ID', 'Barcode_bin',
                     'Barcode_distance_from_start', 'Barcode_distance_from_end',
                     'Start_read_signal', 'Middle_read_signal', 'End_read_signal']))
    
    with open(args.porechop_out, 'rt') as porechop_output:
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

                if read_id not in signals:
                    continue

                barcode_bin = get_final_barcode_call(read_info)
                barcode_bin, start_coord, end_coord = get_start_end_coords(barcode_bin, read_info)
                if barcode_bin is None:
                    continue

                start_signal, middle_signal, end_signal = signals[read_id]

                start_signal_str = ','.join(str(x) for x in start_signal)
                middle_signal_str = ','.join(str(x) for x in middle_signal)
                end_signal_str = ','.join(str(x) for x in end_signal)

                print('\t'.join([read_id, str(barcode_bin), str(start_coord), str(end_coord),
                                 start_signal_str, middle_signal_str, end_signal_str]))


def get_final_barcode_call(read_info):
    assert read_info[-1].startswith('final barcode call:')
    barcode_bin = read_info[-1].split('final barcode call:')[-1].strip()
    assert barcode_bin.startswith('BC') or barcode_bin == 'none'
    if barcode_bin.startswith('BC'):
        return str(int(barcode_bin[2:]))
    else:
        return None


def get_start_barcode_coords(read_info, barcode_bin):
    return get_coords(read_info, barcode_bin, 'start alignments:', 'end alignments:')


def get_end_barcode_coords(read_info, barcode_bin):
    return get_coords(read_info, barcode_bin, 'end alignments:', 'Barcodes:')


def get_coords(read_info, barcode_bin, line_1, line_2):
    i = read_info.index(line_1)
    j = read_info.index(line_2)
    alignment_info = read_info[i+1:j]
    alignment_info = [x for x in alignment_info
                      if x.startswith('Barcode ' + str(barcode_bin) + ' (')][0]
    coords = alignment_info.split('read position: ')[-1]
    return [int(x) for x in coords.split('-')]


def get_end_margin_size(read_info):
    end_seq_line = [x for x in read_info if x.startswith('end: ')][0]
    end_seq = end_seq_line.split('...')[-1]
    end_seq = re.sub('\033.*?m', '', end_seq)  # remove formatting
    return len(end_seq)


def get_start_end_coords(barcode_bin, read_info):
    if barcode_bin:
        try:
            end_margin = get_end_margin_size(read_info)
            start_coord = get_start_barcode_coords(read_info, barcode_bin)[1]
            end_coord = end_margin - get_end_barcode_coords(read_info, barcode_bin)[0]
            return barcode_bin, start_coord, end_coord
        except (IndexError, ValueError):
            return None, '', ''
    else:
        return barcode_bin, '', ''


def get_signal_from_fast5s(fast5_dir, signal_size, stdev_threshold, max_start_end_margin,
                           min_signal_length):
    fast5_files = find_all_fast5s(fast5_dir)
    print('Loading signal from ' + str(len(fast5_files)) + ' fast5 files',
          end='', file=sys.stderr)
    signals = {}
    i = 0

    short_count = 0
    bad_signal_count = 0

    for fast5_file in fast5_files:
        i += 1
        if i % 100 == 0:
            print('.', end='', file=sys.stderr, flush=True)
        with h5py.File(fast5_file, 'r') as hdf5_file:

            read_group = list(hdf5_file['Raw/Reads/'].values())[0]

            read_id = read_group.attrs['read_id'].decode()
            signal = read_group['Signal']

            if len(signal) < min_signal_length:
                short_count += 1
                continue

            bad_signal = False

            # The middle signal is simply the centre-most signal in the read.
            middle_pos = len(signal) // 2
            middle_1 = middle_pos - (signal_size // 2)
            middle_2 = middle_pos + (signal_size // 2)
            middle_signal = signal[middle_1:middle_2]

            start_margin = signal_size * 2
            while True:
                start_signal = signal[:start_margin]
                if numpy.std(start_signal) > stdev_threshold:
                    break
                start_margin += signal_size
                if start_margin > max_start_end_margin:
                    bad_signal = True
                    break
            
            end_margin = signal_size * 2
            while True:
                end_signal = signal[-end_margin:]
                if numpy.std(end_signal) > stdev_threshold:
                    break
                end_margin += signal_size
                if end_margin > max_start_end_margin:
                    bad_signal = True
                    break

            if bad_signal:
                bad_signal_count += 1
                continue

            signals[read_id] = (start_signal, middle_signal, end_signal)

    print(' done', file=sys.stderr)
    print('skipped ' + str(short_count) + ' reads for being too short\n', file=sys.stderr)
    print('skipped ' + str(bad_signal_count) + ' reads for bad signal stdev\n', file=sys.stderr)
    return signals


def find_all_fast5s(directory):
    fast5s = []
    for dir_name, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.fast5'):
                fast5s.append(os.path.join(dir_name, filename))
    return fast5s
