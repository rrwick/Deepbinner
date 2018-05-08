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
        self.captured_output = io.StringIO()
        sys.stderr = self.captured_output

    def tearDown(self):
        sys.stderr = sys.__stderr__  # Reset redirect.

    def test_find_all_fast5s(self):
        fast5s = deepbinner.load_fast5s.find_all_fast5s(self.fast5_dir)
        self.assertEqual(len(fast5s), 5)

    def test_find_all_fast5s_verbose(self):
        fast5s = deepbinner.load_fast5s.find_all_fast5s(self.fast5_dir, verbose=True)
        self.assertEqual(len(fast5s), 5)
        self.assertTrue('Looking for fast5 files' in self.captured_output.getvalue())
        self.assertTrue('5 fast5s found' in self.captured_output.getvalue())

    def test_get_read_id_and_signal_1(self):
        fast5 = self.fast5_dir / '5210_N125509_20170424_FN2002039725_MN19691_sequencing_run_' \
                                 'klebs_033_75349_ch112_read218_strand.fast5'
        read_id, signal = deepbinner.load_fast5s.get_read_id_and_signal(fast5)
        self.assertEqual(read_id, '4de68977-34ab-4c3e-8adf-abfa45d47690')
        self.assertEqual(len(signal), 21962)
        self.assertEqual(signal[0], 380)
        self.assertEqual(signal[9548], 612)

    def test_get_read_id_and_signal_2(self):
        fast5 = self.fast5_dir / '5210_N125509_20170424_FN2002039725_MN19691_sequencing_run_' \
                                 'klebs_033_75349_ch172_read252_strand.fast5'
        read_id, signal = deepbinner.load_fast5s.get_read_id_and_signal(fast5)
        self.assertEqual(read_id, 'baa78ecd-3897-4b96-ad9f-ef3f92f3d2fd')
        self.assertEqual(len(signal), 13725)
        self.assertEqual(signal[0], 528)
        self.assertEqual(signal[13724], 546)

    def test_get_read_id_and_signal_3(self):
        fast5 = self.fast5_dir / 'not_a_real_file.fast5'
        read_id, signal = deepbinner.load_fast5s.get_read_id_and_signal(fast5)
        self.assertEqual(read_id, None)
        self.assertEqual(signal, None)
