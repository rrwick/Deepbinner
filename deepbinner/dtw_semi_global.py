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

import ctypes
import numpy as np
import os
import sys

from numpy.ctypeslib import ndpointer


# To build the C code:
#   g++ -fPIC -O3 -DNDEBUG -Wall -Wextra -pedantic -mtune=native -std=c++14 -c -o dtw.o dtw.cpp
#   g++ -fPIC -O3 -DNDEBUG -Wall -Wextra -pedantic -mtune=native -std=c++14 \
#       -Wl,-install_name,dtw.so -o dtw.so dtw.o -shared -lz
SO_FILE = 'dtw/dtw.so'
SO_FILE_FULL = os.path.join(os.path.dirname(os.path.realpath(__file__)), SO_FILE)
if not os.path.isfile(SO_FILE_FULL):
    sys.exit('Error: could not find ' + SO_FILE)
lib = ctypes.cdll.LoadLibrary(SO_FILE_FULL)


cpp_semi_global_dtw = lib.semi_global_dtw
cpp_semi_global_dtw.restype = ctypes.c_double
cpp_semi_global_dtw.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                ctypes.c_int,
                                ctypes.c_int,
                                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]


def semi_global_dtw(ref, query):
    ref_len = len(ref)
    query_len = len(query)

    alignment = np.empty((ref_len + query_len) * 2, dtype='int32')
    positions = np.empty(2, dtype='int32')
    path_length = np.empty(1, dtype='int32')

    distance = cpp_semi_global_dtw(ref, query, ref_len, query_len, alignment, positions,
                                   path_length)
    path_length = path_length[0]
    alignment = np.resize(alignment, path_length * 2)
    alignment = [(alignment[2*i], alignment[2*i+1]) for i in range(path_length)][::-1]

    return distance, positions[0], positions[1], alignment


def semi_global_dtw_with_rescaling(ref, query):
    """
    Based on this: https://arxiv.org/abs/1705.01620
    """
    query = np.array(query)

    distance, start, end, pairs = 0, 0, 0, []
    overall_slope = 1.0

    iterations = 2

    for i in range(iterations):
        distance, start, end, alignment = semi_global_dtw(ref, query)

        pairs = [(ref[i], query[j]) for i, j in alignment]

        # Don't bother with the linear regression on the last iteration.
        if i == iterations - 1:
            break

        # Do a linear regression on the points between the reference and query.
        x = [p[1] for p in pairs]
        y = [p[0] for p in pairs]
        a = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(a, y)[0]

        # Rescale the query based on the linear regression.
        query = (m * query) + b
        overall_slope *= m

    # If the slope has changed too much, that implies something went wrong!
    if overall_slope < 0.75 or overall_slope > 1.333:
        distance = float('inf')

    return distance, start, end, pairs
