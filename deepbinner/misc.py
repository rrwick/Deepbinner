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

import collections
import sys


def print_summary_table(classifications):
    counts = collections.defaultdict(int)
    for barcode in classifications.values():
        counts[barcode] += 1
    print('', file=sys.stderr)
    print('Barcode     Count', file=sys.stderr)

    number_barcodes, string_barcodes = [], []
    for barcode in counts.keys():
        try:
            number_barcodes.append(int(barcode))
        except ValueError:
            string_barcodes.append(barcode)
    sorted_barcodes = sorted(number_barcodes) + sorted(string_barcodes)

    for barcode in sorted_barcodes:
        print('{:>7} {:>9}'.format(barcode, counts[str(barcode)]), file=sys.stderr)
    print('', file=sys.stderr)
