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

import argparse
import unittest
import deepbinner.classify

def get_args(combine_mode):
    parser = argparse.ArgumentParser()
    parser.add_argument('--require_either', action='store_true')
    parser.add_argument('--require_start', action='store_true')
    parser.add_argument('--require_both', action='store_true')
    return parser.parse_args([combine_mode])


class TestCombineCalls(unittest.TestCase):

    def test_require_either(self):
        args = get_args('--require_either')
        self.assertEqual(deepbinner.classify.combine_calls('4', '4', args), '4')
        self.assertEqual(deepbinner.classify.combine_calls('none', 'none', args), 'none')
        self.assertEqual(deepbinner.classify.combine_calls('5', 'none', args), '5')
        self.assertEqual(deepbinner.classify.combine_calls('none', '7', args), '7')
        self.assertEqual(deepbinner.classify.combine_calls('1', '2', args), 'none')

    def test_require_start(self):
        args = get_args('--require_start')
        self.assertEqual(deepbinner.classify.combine_calls('4', '4', args), '4')
        self.assertEqual(deepbinner.classify.combine_calls('none', 'none', args), 'none')
        self.assertEqual(deepbinner.classify.combine_calls('5', 'none', args), '5')
        self.assertEqual(deepbinner.classify.combine_calls('none', '7', args), 'none')
        self.assertEqual(deepbinner.classify.combine_calls('1', '2', args), 'none')

    def test_require_both(self):
        args = get_args('--require_both')
        self.assertEqual(deepbinner.classify.combine_calls('4', '4', args), '4')
        self.assertEqual(deepbinner.classify.combine_calls('none', 'none', args), 'none')
        self.assertEqual(deepbinner.classify.combine_calls('5', 'none', args), 'none')
        self.assertEqual(deepbinner.classify.combine_calls('none', '7', args), 'none')
        self.assertEqual(deepbinner.classify.combine_calls('1', '2', args), 'none')
