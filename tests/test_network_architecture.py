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

import unittest

from keras.layers import Input
from keras.models import Model
import deepbinner.network_architecture


class TestNetworkArchitecture(unittest.TestCase):

    def test_12_barcodes(self):
        inputs = Input(shape=(1024, 1))
        class_count = 13
        predictions = deepbinner.network_architecture.build_network(inputs, class_count)
        model = Model(inputs=inputs, outputs=predictions)

        self.assertEqual(len(model.layers), 45)
        self.assertEqual(model.count_params(), 107197)
        self.assertEqual(predictions.shape[1], class_count)

    def test_24_barcodes(self):
        inputs = Input(shape=(1024, 1))
        class_count = 25
        predictions = deepbinner.network_architecture.build_network(inputs, class_count)
        model = Model(inputs=inputs, outputs=predictions)

        self.assertEqual(len(model.layers), 45)
        self.assertEqual(model.count_params(), 107785)
        self.assertEqual(predictions.shape[1], class_count)
