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


# Constants for the DTW matrix.
NIL = -1
DIAGONAL = 0
LEFT = 1
UP = 2


def dtw(ref, query):
    """
    Does a semi-global DTW between a reference and query. End gaps are allow in the reference, but
    not in the query, i.e. the query must be contained within the reference.
    """
    ref_len = len(ref)
    query_len = len(query)

    # Initialise the matrices.
    cost_matrix = [[0 for _ in range(query_len)] for _ in range(ref_len)]
    path_matrix = [[0 for _ in range(query_len)] for _ in range(ref_len)]

    # Initialise the top-left corner.
    cost_matrix[0][0] = distance_to(ref[0], query[0])
    path_matrix[0][0] = NIL

    # Initialise the left sides.
    for i in range(1, ref_len):
        cost_matrix[i][0] = cost_matrix[i - 1][0] + distance_to(ref[i], query[0])
        path_matrix[i][0] = UP

    # Initialise the top sides.
    for j in range(1, query_len):
        cost_matrix[0][j] = cost_matrix[0][j - 1] + distance_to(query[j], ref[0])
        path_matrix[0][j] = LEFT

    # Fill in the matrices.
    for i in range(1, ref_len):
        for j in range(1, query_len):
            direction = np.argmin([cost_matrix[i - 1][j - 1],  # 0: diagonal
                                   cost_matrix[i][j - 1],      # 1: left
                                   cost_matrix[i - 1][j]])     # 2: up
            path_matrix[i][j] = direction
            if direction == DIAGONAL:
                res = cost_matrix[i - 1][j - 1]
            elif direction == LEFT:
                res = cost_matrix[i][j - 1]
            elif direction == UP:
                res = cost_matrix[i - 1][j]
            else:
                assert False
            cost_matrix[i][j] = res + distance_to(ref[i], query[j])

    distance = cost_matrix[ref_len - 1][query_len - 1]

    # print()  # TEMP
    # print_cost_matrix(cost_matrix)  # TEMP
    # print()  # TEMP
    # print_path_matrix(path_matrix)  # TEMP
    # print()  # TEMP

    return distance


def distance_to(a, b):
    return (a - b) ** 2


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





# def semi_global_dtw(ref, query):
#     """
#     Does a semi-global DTW between a reference and query. End gaps are allow in the reference, but
#     not in the query, i.e. the query must be contained within the reference.
#     """
#     ref_len = len(ref)
#     query_len = len(query)
#
#     # Initialise the matrices.
#     cost_matrix = [[0 for _ in range(query_len)] for _ in range(ref_len)]
#     path_matrix = [[0 for _ in range(query_len)] for _ in range(ref_len)]
#
#     # Initialise the top-left corner.
#     cost_matrix[0][0] = 0
#     path_matrix[0][0] = NIL
#
#     # Initialise the left sides.
#     for i in range(1, ref_len):
#         cost_matrix[i][0] = 0
#         path_matrix[i][0] = NIL
#
#     # Initialise the top sides.
#     for j in range(1, query_len):
#         cost_matrix[0][j] = cost_matrix[0][j - 1] + distance_to(query[j], ref[0])
#         path_matrix[0][j] = LEFT
#
#     # Fill in the matrices.
#     for i in range(1, ref_len):
#         for j in range(1, query_len):
#             direction = np.argmin([cost_matrix[i - 1][j - 1],  # 0: diagonal
#                                    cost_matrix[i][j - 1],      # 1: left
#                                    cost_matrix[i - 1][j]])     # 2: up
#             path_matrix[i][j] = direction
#             if direction == DIAGONAL:
#                 res = cost_matrix[i - 1][j - 1]
#             elif direction == LEFT:
#                 res = cost_matrix[i][j - 1]
#             elif direction == UP:
#                 res = cost_matrix[i - 1][j]
#             else:
#                 assert False
#
#             # # This penalty serves to discourage the alignment from just cutting across the matrix
#             # # from right to left (which is likely the fastest way through, because the query is
#             # # shorter than the reference, but not what we want).
#             # penalty = 0.0 if direction == DIAGONAL else 0.1
#
#             cost_matrix[i][j] = res + distance_to(ref[i], query[j])
#
#     # The distance is the smallest value on the right hand side (corresponding to the end of the
#     # query sequence).
#     distance = float('inf')
#     ref_end_pos = None
#     for i in range(ref_len):
#         cost = cost_matrix[i][query_len - 1]
#         if cost < distance:
#             distance = cost
#             ref_end_pos = i
#
#     # Trace back until we get to the left side (start of the query sequence).
#     i, j = ref_end_pos, query_len - 1
#     alignment = [(i, j)]
#     while j > 0:
#         if path_matrix[i][j] == DIAGONAL:
#             i, j = i - 1, j - 1
#         elif path_matrix[i][j] == LEFT:
#             j = j - 1
#         elif path_matrix[i][j] == UP:
#             i = i - 1
#         alignment.append((i, j))
#     ref_start_pos = i
#
#     # print()  # TEMP
#     # print_cost_matrix(cost_matrix)  # TEMP
#     # print()  # TEMP
#     # print_path_matrix(path_matrix)  # TEMP
#     # print()  # TEMP
#
#     return distance, ref_start_pos, ref_end_pos, alignment[::-1]


# def print_cost_matrix(cost_matrix):
#     print('\n'.join([' '.join(['{:.1f}'.format(item).rjust(6) for item in row])
#                      for row in cost_matrix]))


# def print_path_matrix(path_matrix):
#     for row in path_matrix:
#         print(' '.join([{-1: 'X', 0: '\\', 1: '-', 2: '|'}[r] for r in row]))
