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
        self.assertEqual(len(fast5s), 7)

    def test_find_all_fast5s_verbose(self):
        fast5s = deepbinner.load_fast5s.find_all_fast5s(self.fast5_dir, verbose=True)
        self.assertEqual(len(fast5s), 7)
        self.assertTrue('Looking for fast5 files' in self.captured_output.getvalue())
        self.assertTrue('7 fast5s found' in self.captured_output.getvalue())

    def test_get_read_id_and_signal_1(self):
        fast5 = self.fast5_dir / '5210_N128870_20180511_FAH70336_MN20200_sequencing_run_057_' \
                                 'Deepbinner_amplicon_43629_read_11206_ch_157_strand.fast5'
        read_id, signal = deepbinner.load_fast5s.get_read_id_and_signal(fast5)
        self.assertEqual(read_id, '177c3867-6812-4476-a6da-9e4d5c43b760')
        self.assertEqual(len(signal), 4971)
        self.assertEqual(signal[0], 714)
        self.assertEqual(signal[4950], 396)

    def test_get_read_id_and_signal_2(self):
        fast5 = self.fast5_dir / '5210_N128870_20180511_FAH70336_MN20200_sequencing_run_057_' \
                                 'Deepbinner_amplicon_43629_read_13863_ch_212_strand.fast5'
        read_id, signal = deepbinner.load_fast5s.get_read_id_and_signal(fast5)
        self.assertEqual(read_id, '9bfcf22c-5654-4b4c-b8f7-d3cebd416338')
        self.assertEqual(len(signal), 4983)
        self.assertEqual(signal[0], 493)
        self.assertEqual(signal[4862], 618)

    def test_get_read_id_and_signal_3(self):
        fast5 = self.fast5_dir / 'not_a_real_file.fast5'
        read_id, signal = deepbinner.load_fast5s.get_read_id_and_signal(fast5)
        self.assertEqual(read_id, None)
        self.assertEqual(signal, None)

    def test_get_read_id_and_signal_4(self):
        fast5 = self.fast5_dir / 'FAK33493_1336eeb8050cb1ca93d41712cf8e817516306473_1000000.fast5'
        read_id, signal = deepbinner.load_fast5s.get_read_id_and_signal(fast5)
        self.assertEqual(read_id, "2fbd86a4-029a-45cf-8f18-411d542572ba")
        self.assertEqual(len(signal), 5395)
        self.assertEqual(signal[0], 505)
        self.assertEqual(signal[5388], 436)


class SingleOrMulti(unittest.TestCase):

    def test_single(self):
        fast5_dir = pathlib.Path(__file__).parent / 'fast5_files'
        fast5s = deepbinner.load_fast5s.find_all_fast5s(fast5_dir)
        self.assertEqual(deepbinner.load_fast5s.determine_single_or_multi_fast5s(fast5s), 'single')

    def test_multi(self):
        fast5_dir = pathlib.Path(__file__).parent / 'multi_read_fast5_files'
        fast5s = deepbinner.load_fast5s.find_all_fast5s(fast5_dir)
        self.assertEqual(deepbinner.load_fast5s.determine_single_or_multi_fast5s(fast5s), 'multi')
