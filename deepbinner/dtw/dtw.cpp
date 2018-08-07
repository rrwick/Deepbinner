// Copyright 2018 Ryan Wick (rrwick@gmail.com)
// https://github.com/rrwick/Nanobuff

// This file is part of Nanobuff. Nanobuff is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later version. Nanobuff is
// distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
// Public License for more details. You should have received a copy of the GNU General Public
// License along with Nanobuff. If not, see <http://www.gnu.org/licenses/>.

#include <algorithm>
#include <limits>

#include "dtw.h"


enum Direction {NIL = -1,
                DIAGONAL = 0,
                LEFT = 1,
                UP = 2};


double distance_to(double a, double b) {
    return (a - b) * (a - b);
}


Direction arg_min_3(double top_left, double left, double top) {

    // If there's a tie, the diagonal direction is preferred.
    if (top_left <= left && top_left <= top)
        return DIAGONAL;
    if (left < top)
        return LEFT;
    if (top < left)
        return UP;

    // If we got here, then it's a tie between going up and going left. Choose one randomly so one
    // isn't preferred over the other.
    if (rand() % 2 == 0)
        return LEFT;
    else
        return UP;
}

int matrix_i(int i, int j, int query_len) {
    return (i * query_len) + j;
}


// This function carries out a straightforward dynamic time warping alignment between two
// sequences: ref and seq.
// The result is stored in the alignment variable as pairs of indices, in reverse order (from end
// of sequence to start). E.g. if alignment contains [5, 5, 4, 5, 3, 4, 3, 3, 2, 2, 1, 1, 0, 0],
// that indicates these pairs: (5, 5), (4, 5), (3, 4), (3, 3), (2, 2), (1, 1), (0, 0), where the
// first number in each pair contains the seq index and the second contains the ref index.
double semi_global_dtw(const double * ref, const double * query, int ref_len, int query_len,
                       int * alignment, int * positions, int * path_length) {

    Direction direction;
    int index, diag_i, top_i, left_i;

    // Initialise the matrices. Instead of actually building a 2D arrays, I build them as 1D arrays
    // and just index in as if they were 2D, which seems a bit faster.
    int array_size = ref_len * query_len;
    auto cost_matrix = new double[array_size];
    auto path_matrix = new Direction[array_size];

    // Initialise the top-left corners.
    cost_matrix[0] = 0;
    path_matrix[0] = NIL;

    // Initialise the left sides.
    for (int i = 1; i < ref_len; ++i) {
        index = matrix_i(i, 0, query_len);
        cost_matrix[index] = 0;
        path_matrix[index] = NIL;
    }

    // Initialise the top sides.
    for (int j = 1; j < query_len; ++j) {
        index = matrix_i(0, j, query_len);
        left_i = matrix_i(0, j-1, query_len);
        cost_matrix[index] = cost_matrix[left_i] + distance_to(query[j], ref[0]);
        path_matrix[index] = LEFT;
    }

    // Fill in the matrices.
    double res = 0.0;
    for (int i = 1; i < ref_len; ++i) {
        for (int j = 1; j < query_len; ++j) {
            index = matrix_i(i, j, query_len);
            left_i = matrix_i(i, j-1, query_len);
            top_i = matrix_i(i-1, j, query_len);
            diag_i = matrix_i(i-1, j-1, query_len);

            direction = arg_min_3(cost_matrix[diag_i],
                                  cost_matrix[left_i],
                                  cost_matrix[top_i]);
            path_matrix[index] = direction;

            if (direction == DIAGONAL)
                res = cost_matrix[diag_i];
            else if (direction == LEFT)
                res = cost_matrix[left_i];
            else if (direction == UP)
                res = cost_matrix[top_i];
            cost_matrix[index] = res + distance_to(ref[i], query[j]);
        }
    }

    double distance = std::numeric_limits<double>::max();
    int ref_end_pos = 0;
    int j = query_len - 1;  // end of query
    for (int i = 1; i < ref_len; ++i) {
        double cost = cost_matrix[matrix_i(i, j, query_len)];
        if (cost < distance) {
            distance = cost;
            ref_end_pos = i;
        }
    }

    int final_path_length = 0;
    int i = ref_end_pos;
    for (int a = 0; ; ++a) {
        alignment[a*2] = i;
        alignment[a*2 + 1] = j;
        ++final_path_length;
        if (j == 0)
            break;
        index = matrix_i(i, j, query_len);
        if (path_matrix[index] == DIAGONAL) {
            --i; --j;
        }
        else if (path_matrix[index] == LEFT) {
            --j;
        }
        else if (path_matrix[index] == UP) {
            --i;
        }
    }

    path_length[0] = final_path_length;
    positions[0] = i;
    positions[1] = ref_end_pos;

    delete[] cost_matrix;
    delete[] path_matrix;
    return distance;
}
