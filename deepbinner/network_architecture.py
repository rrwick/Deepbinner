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

import random
from keras.layers import Dense, Conv1D, MaxPooling1D, AveragePooling1D, Dropout, Flatten, \
    concatenate, BatchNormalization, GaussianNoise, GlobalAveragePooling1D, Softmax
from keras.models import Model


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


def build_random_network(inputs, class_count):
    """
    This function (and the ones that follow) builds a random network architecture. It was used to
    screen hundreds of possible architectures to home in on a good one.
    """
    max_allowed_parameters = 250000

    while True:
        try:
            x = build_random_network_2(inputs, class_count)
            params = Model(inputs=inputs, outputs=x).count_params()
            if params <= max_allowed_parameters:
                break
            else:
                print('\nTOO MANY PARAMETERS ({})\nTRYING AGAIN\n\n\n\n'.format(params))
        except ValueError:
            print('\nFAILED DUE TO SMALL DATA DIMENSION\nTRYING AGAIN\n\n\n\n')
            pass
    return x


def build_random_network_2(inputs, class_count):
    print('Building random network:')
    x = inputs

    print()
    print('# shape = ' + str(x.shape))

    # Maybe add a noise layer right to the beginning (essentially as a form of data augmentation).
    if random.random() < 0.5:
        x = add_noise_layer(x, random.uniform(0.0, 0.05))

    # Maybe add an average pooling layer. If strides == 1, this will just serve to smooth the data.
    # If strides > 1, this will downscale the data too.
    if random.random() < 0.25:
        print()
        pool_size = random.randint(2, 3)
        strides = random.randint(1, pool_size)
        x = add_average_pooling_layer(x, pool_size, strides)

    # Decide on a starting number of filters. This will grow over the convolutional groups
    filters = random.randint(8, 48)

    max_pooling_count, average_pooling_count, stride_count = 0, 0, 0
    parallel_module_count, serial_module_count = 0, 0
    normalisation_count, bottleneck_count = 0, 0
    serial_module_layer_count, kernel_size_sum = 0, 0
    while True:
        need_to_pool = True

        # Add a convolutional group that is either a parallel (inception-like) module...
        if random.random() < 0.25:
            bottleneck_filters = random.randint(4, filters)
            x = add_parallel_module(x, filters, bottleneck_filters)
            parallel_module_count += 1

        # ...or just a stack of convolutional layers.
        else:
            conv_count = random.choice([1, 2, 3, 4])
            serial_module_count += 1
            serial_module_layer_count += conv_count
            print()
            print('# Convolutional group of {} layer{}'.format(conv_count,
                                                               '' if conv_count == 1 else 's'))
            for i in range(conv_count):
                kernel_size = random.choice([3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9])
                kernel_size_sum += kernel_size

                # The last convolutional layer in the group may get a stride of more than one which
                # will serve to reduce the data's dimensions (in place of a pooling layer).
                if i == conv_count - 1:
                    stride = random.choice([1, 1, 1, 2, 2, 3])
                else:
                    stride = 1
                if stride == 1:
                    dilation = random.choice([1, 1, 1, 1, 1, 1, 2, 2, 3])
                else:
                    dilation = 1
                    need_to_pool = False
                    stride_count += 1
                x = add_conv_layer(x, filters, kernel_size, stride, dilation)

        # If we didn't reduce the data's dimension with a stride, do so now with pooling.
        if need_to_pool:
            pool_size = random.choice([2, 2, 3])
            if random.random() < 0.9:
                x = add_max_pooling_layer(x, pool_size)
                max_pooling_count += 1
            else:
                x = add_average_pooling_layer(x, pool_size)
                average_pooling_count += 1

        # Possibly add one or more optional layers after the convolutional group.
        if random.random() < 0.25:
            x = add_normalization_layer(x)
            normalisation_count += 1
        if random.random() < 0.66667:
            x = add_dropout_layer(x, random.uniform(0.0, 0.15))
        if random.random() < 0.33333:
            x = add_noise_layer(x, random.uniform(0.0, 0.05))
        if random.random() < 0.2:
            bottleneck_filters = random.randint(4, int(filters * 0.8))
            x = add_bottleneck_layer(x, bottleneck_filters)
            bottleneck_count += 1

        # We continue adding convolutional groups until the dimensions are sufficiently small.
        dimension = int(x.shape[1])
        if random.uniform(0.0, dimension) < 5.0:
            break

        # Increase the filters for the next round.
        filters = int(filters * random.uniform(1.25, 2.5))

    print()
    dimension_before_finishing = int(x.shape[1])

    # Finish the network in two possible ways:
    # option 1: the traditional method of flattening followed by dense layers
    dense_count = 0
    if random.random() < 0.0:
        print('# Finishing with dense layers')
        x = add_flatten_layer(x)

        print()
        print('# Fully connected layers')
        for _ in range(random.randint(0, 2)):
            count = int(random.uniform(3, 15) ** 2)
            x = add_dense_layer(x, count)
            dense_count += 1

        # Maybe add a final dropout layer.
        if random.random() < 0.66667:
            x = add_dropout_layer(x, random.uniform(0.0, 0.2))

        print()
        print('# Final layer to output classes')
        x = add_final_dense_layer(x, class_count)
        dense_count += 1

    # option 2: global average pooling directly to the output classes
    else:
        print('# Finishing with global average pooling')
        print("x = Conv1D(filters={}, kernel_size=1, activation='relu')(x)".format(class_count))
        x = Conv1D(filters=class_count, kernel_size=1, activation='relu')(x)
        print('# shape = ' + str(x.shape))
        print('x = GlobalAveragePooling1D()(x)')
        x = GlobalAveragePooling1D()(x)
        print('x = Softmax()(x)')
        x = Softmax()(x)
        print('# shape = ' + str(x.shape))

    try:
        mean_serial_kernel_size = kernel_size_sum / serial_module_layer_count
    except ZeroDivisionError:
        mean_serial_kernel_size = 0

    print('\nModel details:')
    print('\t'.join(['Parallel_module_count',
                     'Serial_module_count',
                     'Serial_module_layer_count',
                     'Mean_serial_kernel_size',
                     'Max_pooling_count',
                     'Average_pooling_count',
                     'Stride_count',
                     'Normalisation_count',
                     'Bottleneck_count',
                     'Dimension_before_finishing',
                     'Dense_layer_count']))
    print('\t'.join(str(x) for x in [parallel_module_count,
                                     serial_module_count,
                                     serial_module_layer_count,
                                     mean_serial_kernel_size,
                                     max_pooling_count,
                                     average_pooling_count,
                                     stride_count,
                                     normalisation_count,
                                     bottleneck_count,
                                     dimension_before_finishing,
                                     dense_count]))
    print()

    return x


def add_dense_layer(x, count):
    print("x = Dense({}, activation='relu')(x)".format(count))
    x = Dense(count, activation='relu')(x)
    print('# shape = ' + str(x.shape))
    return x


def add_final_dense_layer(x, class_count):
    print("x = Dense({}, activation='softmax')(x)".format(class_count))
    x = Dense(class_count, activation='softmax')(x)
    print('# shape = ' + str(x.shape))
    return x


def add_flatten_layer(x):
    print('x = Flatten()(x)')
    x = Flatten()(x)
    print('# shape = ' + str(x.shape))
    return x


def add_noise_layer(x, noise_level):
    print()
    print('x = GaussianNoise(stddev={})(x)'.format(noise_level))
    x = GaussianNoise(stddev=noise_level)(x)
    return x


def add_normalization_layer(x):
    print()
    print('x = BatchNormalization()(x)')
    x = BatchNormalization()(x)
    return x


def add_dropout_layer(x, dropout_frac):
    print()
    print('x = Dropout(rate={})(x)'.format(dropout_frac))
    x = Dropout(rate=dropout_frac)(x)
    return x


def add_max_pooling_layer(x, pool_size):
    print('x = MaxPooling1D(pool_size={})(x)'.format(pool_size))
    x = MaxPooling1D(pool_size=pool_size)(x)
    print('# shape = ' + str(x.shape))
    return x


def add_average_pooling_layer(x, pool_size, strides=None):
    if strides is None:
        print('x = AveragePooling1D(pool_size={})(x)'.format(pool_size))
        x = AveragePooling1D(pool_size=pool_size)(x)
    else:
        print('x = AveragePooling1D(pool_size={}, strides={})(x)'.format(pool_size, strides))
        x = AveragePooling1D(pool_size=pool_size, strides=strides)(x)
    print('# shape = ' + str(x.shape))
    return x


def add_bottleneck_layer(x, filters):
    print()
    print('# Bottleneck')
    print("x = Conv1D(filters={}, kernel_size=1, activation='relu')(x)".format(filters))
    x = Conv1D(filters=filters, kernel_size=1, activation='relu')(x)
    print('# shape = ' + str(x.shape))
    return x


def add_conv_layer(x, filters, kernel_size, strides, dilation_rate):
    print("x = Conv1D(filters={}, kernel_size={}, strides={}, dilation_rate={}, activation='relu')"
          "(x)".format(filters, kernel_size, strides, dilation_rate))
    x = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides,
               dilation_rate=dilation_rate, activation='relu')(x)
    print('# shape = ' + str(x.shape))
    return x


def add_parallel_module(x, conv_filters, bottleneck_filters):
    """
    Loosely based on the inception descriptions given here:
        https://towardsdatascience.com/neural-network-architectures-156e5bad51ba
    And illustrated here:
        https://cdn-images-1.medium.com/max/1600/0*SJ7DP_-0R1vdpVzv.jpg
    """
    print()
    print('# Parallel module')

    # This parallel module will include 2, 3 or 4 parts.
    parts_to_include = set(random.sample([1, 2, 3, 4], random.randint(2, 4)))
    parts_to_concatenate = []
    parts_to_concatenate_str = []

    if 1 in parts_to_include:
        print("x1 = AveragePooling1D(pool_size=3, strides=1, padding='same')(x)")
        x1 = AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
        print("x1 = Conv1D(filters={}, kernel_size=1, padding='same', "
              "activation='relu')(x1)".format(conv_filters))
        x1 = Conv1D(filters=conv_filters, kernel_size=1, padding='same', activation='relu')(x1)
        parts_to_concatenate.append(x1)
        parts_to_concatenate_str.append('x1')

    if 2 in parts_to_include:
        print("x2 = Conv1D(filters={}, kernel_size=1, padding='same', "
              "activation='relu')(x)".format(conv_filters))
        x2 = Conv1D(filters=conv_filters, kernel_size=1, padding='same', activation='relu')(x)
        parts_to_concatenate.append(x2)
        parts_to_concatenate_str.append('x2')

    if 3 in parts_to_include:
        print("x3 = Conv1D(filters={}, kernel_size=1, padding='same', "
              "activation='relu')(x)".format(bottleneck_filters))
        x3 = Conv1D(filters=bottleneck_filters, kernel_size=1, padding='same', activation='relu')(x)
        print("x3 = Conv1D(filters={}, kernel_size=3, padding='same', "
              "activation='relu')(x3)".format(conv_filters))
        x3 = Conv1D(filters=conv_filters, kernel_size=3, padding='same', activation='relu')(x3)
        parts_to_concatenate.append(x3)
        parts_to_concatenate_str.append('x3')

    if 4 in parts_to_include:
        print("x4 = Conv1D(filters={}, kernel_size=1, padding='same', "
              "activation='relu')(x)".format(bottleneck_filters))
        x4 = Conv1D(filters=bottleneck_filters, kernel_size=1, padding='same', activation='relu')(x)
        print("x4 = Conv1D(filters={}, kernel_size=5, padding='same', "
              "activation='relu')(x4)".format(conv_filters))
        x4 = Conv1D(filters=conv_filters, kernel_size=5, padding='same', activation='relu')(x4)
        parts_to_concatenate.append(x4)
        parts_to_concatenate_str.append('x4')

    print('x = concatenate([{}], axis=2)'.format(', '.join(parts_to_concatenate_str)))
    x = concatenate(parts_to_concatenate, axis=2)

    return x
