// Copyright 2018 Ryan Wick (rrwick@gmail.com)
// https://github.com/rrwick/Nanobuff

// This file is part of Nanobuff. Nanobuff is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later version. Nanobuff is
// distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
// Public License for more details. You should have received a copy of the GNU General Public
// License along with Nanobuff. If not, see <http://www.gnu.org/licenses/>.

#ifndef DTW_H
#define DTW_H

// Functions that are called by the Python script must have C linkage, not C++ linkage.
extern "C" {
    double semi_global_dtw(const double * ref, const double * query, int ref_len, int query_len,
                           int * alignment, int * positions, int * path_length);
}


#endif  // DTW_H
