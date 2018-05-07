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

from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Dropout, concatenate,\
    BatchNormalization, GaussianNoise, GlobalAveragePooling1D, Softmax


def build_network(inputs, class_count):
    """
    This function builds the standard network used in Deepbinner.
    """
    x = inputs

    # Add some noise to augment the training data.
    x = GaussianNoise(stddev=0.02)(x)

    # Conv layer with stride of 2 (halves the size)
    x = Conv1D(filters=48, kernel_size=3, strides=2, padding='same', activation='relu')(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.15)(x)

    # Conv group: 3 layers of 3-kernels
    x = Conv1D(filters=48, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(filters=48, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(filters=48, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.15)(x)

    # Bottleneck down to 16 filters (reduces the number of parameters a bit)
    x = Conv1D(filters=16, kernel_size=1, activation='relu')(x)

    # Conv group: 2 layers of 3-kernels
    x = Conv1D(filters=48, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(filters=48, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.15)(x)

    # Conv group: 2 layers of 3-kernels
    x = Conv1D(filters=48, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(filters=48, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.15)(x)

    # Inception-style group
    x1 = AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
    x1 = Conv1D(filters=48, kernel_size=1, padding='same', activation='relu')(x1)
    x2 = Conv1D(filters=48, kernel_size=1, padding='same', activation='relu')(x)
    x3 = Conv1D(filters=16, kernel_size=1, padding='same', activation='relu')(x)
    x3 = Conv1D(filters=48, kernel_size=3, padding='same', activation='relu')(x3)
    x4 = Conv1D(filters=16, kernel_size=1, padding='same', activation='relu')(x)
    x4 = Conv1D(filters=48, kernel_size=3, padding='same', activation='relu')(x4)
    x4 = Conv1D(filters=48, kernel_size=3, padding='same', activation='relu')(x4)
    x = concatenate([x1, x2, x3, x4], axis=2)
    x = MaxPooling1D(pool_size=2)(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.15)(x)

    # Conv layer with stride of 2 (halves the size)
    x = Conv1D(filters=48, kernel_size=3, strides=2, activation='relu', padding='same')(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.15)(x)

    # Conv group: 2 layers of 3-kernels
    x = Conv1D(filters=48, kernel_size=3, activation='relu', padding='same')(x)
    x = Conv1D(filters=48, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.15)(x)

    # Finish with a global average pooling approach (no fully connected layers)
    x = Conv1D(filters=class_count, kernel_size=1, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Softmax()(x)

    return x
