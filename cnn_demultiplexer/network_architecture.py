
import random
from keras.layers import Dense, Conv1D, MaxPooling1D, AveragePooling1D, Dropout, Flatten, \
    concatenate, BatchNormalization, GaussianNoise
from keras.models import Model


def build_random_network(inputs, class_count):
    max_allowed_parameters = 500000

    while True:
        try:
            x = build_random_network_2(inputs, class_count)
            params = Model(inputs=inputs, outputs=x).count_params()
            if params <= max_allowed_parameters:
                break
            else:
                print('\nTOO MANY PARAMETERS ({})\nTRYING AGAIN\n\n\n'.format(params))
        except ValueError:
            print('\nFAILED DUE TO SMALL DATA DIMENSION\nTRYING AGAIN\n\n\n')
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

    while True:
        need_to_pool = True

        # Add a convolutional group that is either an parallel (inception-like) module...
        if random.random() < 0.25:
            bottleneck_filters = random.randint(4, filters)
            x = add_parallel_module(x, filters, bottleneck_filters)

        # ...or just a stack of convolutional layers.
        else:
            conv_count = random.choice([1, 2, 3, 4])
            print()
            print('# Convolutional group of {} layer{}'.format(conv_count,
                                                               '' if conv_count == 1 else 's'))
            for _ in range(conv_count):
                kernel_size = random.choice([3, 3, 3, 3, 3, 5, 5, 5, 7, 9])
                stride = random.choice([1, 1, 1, 1, 1, 1, 2, 2, 3])
                if stride == 1:
                    dilation = random.choice([1, 1, 1, 1, 1, 1, 2, 2, 3])
                else:
                    dilation = 1
                    need_to_pool = False
                x = add_conv_layer(x, filters, kernel_size, stride, dilation)

        # Reduce the data's dimension by pooling (only necessary if we didn't use a higher stride).
        if need_to_pool:
            pool_size = random.randint(2, 3)
            if random.random() < 0.8:
                x = add_max_pooling_layer(x, pool_size)
            else:
                x = add_average_pooling_layer(x, pool_size)

        if random.random() < 0.33333:
            x = add_normalization_layer(x)

        if random.random() < 0.66667:
            x = add_dropout_layer(x, random.uniform(0.0, 0.15))

        if random.random() < 0.33333:
            x = add_noise_layer(x, random.uniform(0.0, 0.05))

        if random.random() < 0.33333:
            bottleneck_filters = random.randint(4, int(filters * 0.8))
            x = add_bottleneck_layer(x, bottleneck_filters)

        # We continue adding convolutional groups until the dimensions are sufficiently small.
        dimension = x.shape[1]
        if dimension < 5:
            break
        elif dimension < 10:
            if random.random() < 0.75:
                break
        elif dimension < 20:
            if random.random() < 0.5:
                break
        elif dimension < 30:
            if random.random() < 0.25:
                break

        # Increase the filters for the next round.
        filters = int(filters * random.uniform(1.25, 2.5))

    print()
    x = add_flatten_layer(x)

    # Add some fully connected layers.
    print()
    print('# Fully connected layers')
    for _ in range(random.randint(1, 3)):
        count = int(random.uniform(3, 15) ** 2)
        x = add_dense_layer(x, count)

    # Maybe add a final dropout layer.
    if random.random() < 0.66667:
        x = add_dropout_layer(x, random.uniform(0.0, 0.2))

    print()
    print('# Final layer to output classes')
    x = add_final_dense_layer(x, class_count)

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


def random_80(inputs, class_count):
    """
    This was the best performing network in my second batch of randomly-generated architectures.
    """
    x = inputs

    x = Conv1D(filters=42, kernel_size=3, strides=2, activation='relu')(x)

    x = Dropout(rate=0.0860010780012494)(x)

    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(rate=0.046698083895803544)(x)

    x = Conv1D(filters=14, kernel_size=1, activation='relu')(x)

    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.13881098500911096)(x)

    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.07401793205106891)(x)

    x1 = AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
    x1 = Conv1D(filters=42, kernel_size=1, padding='same', activation='relu')(x1)
    x2 = Conv1D(filters=42, kernel_size=1, padding='same', activation='relu')(x)
    x3 = Conv1D(filters=13, kernel_size=1, padding='same', activation='relu')(x)
    x3 = Conv1D(filters=42, kernel_size=3, padding='same', activation='relu')(x3)
    x4 = Conv1D(filters=13, kernel_size=1, padding='same', activation='relu')(x)
    x4 = Conv1D(filters=42, kernel_size=3, padding='same', activation='relu')(x4)
    x4 = Conv1D(filters=42, kernel_size=3, padding='same', activation='relu')(x4)
    x = concatenate([x1, x2, x3, x4], axis=2)
    x = MaxPooling1D(pool_size=2)(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.17179247399043648)(x)

    x = Conv1D(filters=42, kernel_size=3, strides=2, activation='relu')(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.12225393209569824)(x)

    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.16411133823036847)(x)

    x = Conv1D(filters=6, kernel_size=1, activation='relu')(x)

    x = Flatten()(x)
    x = Dense(106, activation='relu')(x)
    x = Dense(150, activation='relu')(x)
    x = Dense(30, activation='relu')(x)
    x = Dense(244, activation='relu')(x)

    x = Dense(class_count, activation='softmax')(x)

    return x
