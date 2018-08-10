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

import sys

from .trim_signal import normalise
from .dtw_semi_global import semi_global_dtw_with_rescaling
from .prep_functions import align_read_to_reference, align_adapter_to_read_seq, trim_signal, \
    get_best_barcode, align_barcode_to_read_dtw, get_training_sample_around_signal, \
    get_training_sample_from_middle_of_signal, get_training_sample_before_signal
from . import sequences
from . import signals


def prep_rapid_read_start():
     pass
