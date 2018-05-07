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

import io
import pathlib
import sys
import unittest

import deepbinner.load_fast5s


class TestLoadFast5s(unittest.TestCase):

    def setUp(self):
        self.fast5_dir = pathlib.Path(__file__).parent / 'fast5_files'

    def test_find_all_fast5s(self):
        fast5s = deepbinner.load_fast5s.find_all_fast5s(self.fast5_dir)
        self.assertEqual(len(fast5s), 10)

    def test_find_all_fast5s_verbose(self):
        captured_output = io.StringIO()  # Create StringIO object
        sys.stderr = captured_output  # and redirect stdout.
        fast5s = deepbinner.load_fast5s.find_all_fast5s(self.fast5_dir, verbose=True)
        sys.stderr = sys.__stdout__  # Reset redirect.

        self.assertEqual(len(fast5s), 10)
        self.assertTrue('Looking for fast5 files' in captured_output.getvalue())
        self.assertTrue('10 fast5s found' in captured_output.getvalue())

    def test_get_read_id_and_signal_1(self):
        fast5 = self.fast5_dir / '5210_N125509_20170424_FN2002039725_MN19691_sequencing_run_' \
                                 'klebs_033_75349_ch1_read172_strand.fast5'
        read_id, signal = deepbinner.load_fast5s.get_read_id_and_signal(fast5)
        self.assertEqual(read_id, 'be7bafdb-724c-4f97-8b79-1c08bf334c98')
        self.assertEqual(len(signal), 37641)
        self.assertEqual(signal[0], 300)
        self.assertEqual(signal[15460], 494)

    def test_get_read_id_and_signal_2(self):
        fast5 = self.fast5_dir / '5210_N125509_20170424_FN2002039725_MN19691_sequencing_run_' \
                                 'klebs_033_75349_ch1_read181_strand.fast5'
        read_id, signal = deepbinner.load_fast5s.get_read_id_and_signal(fast5)
        self.assertEqual(read_id, 'db382b1e-4dd2-44a7-a980-1d1db543aade')
        self.assertEqual(len(signal), 56670)
        self.assertEqual(signal[0], 339)
        self.assertEqual(signal[56330], 329)

    def test_get_read_id_and_signal_3(self):
        fast5 = self.fast5_dir / 'not_a_real_file.fast5'
        read_id, signal = deepbinner.load_fast5s.get_read_id_and_signal(fast5)
        self.assertEqual(read_id, None)
        self.assertEqual(signal, None)
