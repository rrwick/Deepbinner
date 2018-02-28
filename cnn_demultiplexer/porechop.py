

import re
from .load_fast5s import get_signal_from_fast5s


def training_data_from_porechop(args):
    signals = get_signal_from_fast5s(args.fast5_dir, args.signal_size, args.max_start_end_margin,
                                     args.min_signal_length)

    print('\t'.join(['Read_ID', 'Barcode_bin',
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

                print('\t'.join([read_id, str(barcode_bin),
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
