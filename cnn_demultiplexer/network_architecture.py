
import random
from keras.layers import Dense, Conv1D, MaxPooling1D, AveragePooling1D, Dropout, Flatten, \
    concatenate, BatchNormalization, GaussianNoise


def classic_cnn(inputs, class_count):
    """
    This architecture follows a straightforward CNN construction: convolution layers followed by
    max pooling, then some fully connected layers. The number of filters grows as the data is
    reduced in size.
    """
    x = double_3_conv_with_max_pooling(inputs, filters=16, pool_size=2)
    x = Dropout(0.25)(x)
    x = double_3_conv_with_max_pooling(x, filters=32, pool_size=2)
    x = Dropout(0.25)(x)
    x = double_3_conv_with_max_pooling(x, filters=64, pool_size=2)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.25)(x)
    return Dense(class_count, activation='softmax')(x)


def classic_cnn_with_bottlenecks(inputs, class_count):
    """
    This architecture is like classic_cnn, but with bottleneck 1 convolutions between the groups.
    This significantly reduces the number of parameters in the model.
    """
    x = double_3_conv_with_max_pooling(inputs, filters=16, pool_size=2)
    x = Dropout(0.25)(x)
    x = bottleneck(x, filters=8)
    x = double_3_conv_with_max_pooling(x, filters=32, pool_size=2)
    x = Dropout(0.25)(x)
    x = bottleneck(x, filters=16)
    x = double_3_conv_with_max_pooling(x, filters=64, pool_size=2)
    x = Dropout(0.25)(x)
    x = bottleneck(x, filters=16)
    x = Flatten()(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.25)(x)
    return Dense(class_count, activation='softmax')(x)


def inception_network(inputs, class_count):
    """
    This architecture is loosely based on the ones in these papers:
        https://arxiv.org/abs/1409.4842
        https://arxiv.org/abs/1512.00567
    """
    x = single_3_conv_with_2_stride(inputs, filters=16)
    x = double_3_conv_with_max_pooling(x, filters=24, pool_size=2)
    x = Dropout(0.25)(x)
    x = double_3_conv_with_max_pooling(x, filters=32, pool_size=2)
    x = Dropout(0.25)(x)
    x = inception_module_with_max_pooling(x, conv_filters=48, bottleneck_filters=24, pool_size=2)
    x = Dropout(0.25)(x)
    x = inception_module_with_max_pooling(x, conv_filters=64, bottleneck_filters=32, pool_size=2)
    x = Dropout(0.25)(x)
    x = inception_module_with_max_pooling(x, conv_filters=96, bottleneck_filters=48, pool_size=2)
    x = Dropout(0.25)(x)
    x = inception_module_with_max_pooling(x, conv_filters=128, bottleneck_filters=64, pool_size=2)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.25)(x)
    return Dense(class_count, activation='softmax')(x)


def build_random_network(inputs, class_count):
    print('Building random network:')
    x = inputs

    # Add the convolutional layers, which reduce the size of the data but add filters.
    filters = random.randint(8, 48)
    for _ in range(random.randint(3, 8)):
        remaining_dims = x.shape[1]

        # Randomly choose one of the dimension-reducing layers/groups.
        group_type = random.choice(['stride', 'double_max', 'triple_max', 'inception'])
        if group_type == 'stride' and remaining_dims >= 3:
            print('single_3_conv_with_2_stride(x, filters={})'.format(filters))
            x = single_3_conv_with_2_stride(x, filters=filters)
        if group_type == 'double_max' and remaining_dims >= 6:
            print('double_3_conv_with_max_pooling(x, filters={}, pool_size=2)'.format(filters))
            x = double_3_conv_with_max_pooling(x, filters=filters, pool_size=2)
        if group_type == 'triple_max' and remaining_dims >= 8:
            print('triple_3_conv_with_max_pooling(x, filters={}, pool_size=2)'.format(filters))
            x = triple_3_conv_with_max_pooling(x, filters=filters, pool_size=2)
        if group_type == 'inception':
            bottleneck_filters = random.randint(2, filters)
            print('inception_module_with_max_pooling(x, conv_filters={}, bottleneck_filters={}, '
                  'pool_size=2)'.format(filters, bottleneck_filters))
            x = inception_module_with_max_pooling(x, conv_filters=filters,
                                                  bottleneck_filters=bottleneck_filters,
                                                  pool_size=2)
        if random.random() < 0.33333:
            # Add a batch normalization layer.
            dropout_frac = random.uniform(0.0, 0.5)
            print('BatchNormalization()(x)'.format(dropout_frac))
            x = BatchNormalization()(x)

        if random.random() < 0.66667:
            # Add a dropout layer.
            dropout_frac = random.uniform(0.0, 0.2)
            print('Dropout(rate={})(x)'.format(dropout_frac))
            x = Dropout(rate=dropout_frac)(x)

        if random.random() < 0.33333:
            # Add a noise layer.
            noise_level = random.uniform(0.0, 0.1)
            print('GaussianNoise(stddev={})(x)'.format(noise_level))
            x = GaussianNoise(stddev=noise_level)(x)

        if random.random() < 0.33333:
            # Add a bottleneck layer.
            bottleneck_filters = random.randint(2, filters)
            print('bottleneck(x, filters={})'.format(bottleneck_filters))
            x = bottleneck(x, filters=bottleneck_filters)

        # Increase the filters for the next round.
        filters *= int(random.uniform(1.0, 2.5))

    print('Flatten()(x)')
    x = Flatten()(x)

    # Add some fully connected layers.
    for _ in range(random.randint(1, 4)):
        count = int(random.uniform(2, 20) ** 2)
        print("Dense({}, activation='relu')(x)".format(count))
        x = Dense(count, activation='relu')(x)

    if random.random() < 0.66667:
        # Add a dropout layer.
        dropout_frac = random.uniform(0.0, 0.2)
        print('Dropout(rate={})(x)'.format(dropout_frac))
        x = Dropout(rate=dropout_frac)(x)

    # Connect to the final classes.
    print("Dense({}, activation='softmax')(x)".format(class_count))
    x = Dense(class_count, activation='softmax')(x)

    print()
    return x


def single_3_conv_with_2_stride(inputs, filters):
    x = Conv1D(filters=filters, kernel_size=3, strides=2, activation='relu')(inputs)
    return x


def double_3_conv_with_max_pooling(inputs, filters, pool_size):
    x = Conv1D(filters=filters, kernel_size=3, activation='relu')(inputs)
    x = Conv1D(filters=filters, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    return x


def triple_3_conv_with_max_pooling(inputs, filters, pool_size):
    x = Conv1D(filters=filters, kernel_size=3, activation='relu')(inputs)
    x = Conv1D(filters=filters, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=filters, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    return x


def inception_module_with_max_pooling(inputs, conv_filters, bottleneck_filters, pool_size):
    """
    Based on the 'Inception V3' description given here:
        https://towardsdatascience.com/neural-network-architectures-156e5bad51ba
    And illustrated here:
        https://cdn-images-1.medium.com/max/1600/0*SJ7DP_-0R1vdpVzv.jpg
    """
    x1 = AveragePooling1D(pool_size=3, strides=1, padding='same')(inputs)
    x1 = Conv1D(filters=conv_filters, kernel_size=1, padding='same', activation='relu')(x1)

    x2 = Conv1D(filters=conv_filters, kernel_size=1, padding='same', activation='relu')(inputs)

    x3 = Conv1D(filters=bottleneck_filters, kernel_size=1, padding='same', activation='relu')(inputs)
    x3 = Conv1D(filters=conv_filters, kernel_size=3, padding='same', activation='relu')(x3)

    x4 = Conv1D(filters=bottleneck_filters, kernel_size=1, padding='same', activation='relu')(inputs)
    x4 = Conv1D(filters=conv_filters, kernel_size=3, padding='same', activation='relu')(x4)
    x4 = Conv1D(filters=conv_filters, kernel_size=3, padding='same', activation='relu')(x4)

    x = concatenate([x1, x2, x3, x4], axis=2)

    x = MaxPooling1D(pool_size=pool_size)(x)
    return x


def bottleneck(inputs, filters):
    x = Conv1D(filters=filters, kernel_size=1, activation='relu')(inputs)
    return x
