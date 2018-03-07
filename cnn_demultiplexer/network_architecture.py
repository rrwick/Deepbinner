
import random
from keras.layers import Dense, Conv1D, MaxPooling1D, AveragePooling1D, Dropout, Flatten, \
    concatenate, BatchNormalization, GaussianNoise, GlobalAveragePooling1D, Softmax
from keras.models import Model


def build_random_network(inputs, class_count):
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


def random_080(inputs, class_count):
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


def random_339(inputs, class_count):
    x = inputs

    x = GaussianNoise(stddev=0.013397584943997)(x)

    x = Conv1D(filters=21, kernel_size=3, strides=1, dilation_rate=1, activation='relu')(x)
    x = Conv1D(filters=21, kernel_size=9, strides=1, dilation_rate=1, activation='relu')(x)
    x = Conv1D(filters=21, kernel_size=3, strides=1, dilation_rate=2, activation='relu')(x)
    x = Conv1D(filters=21, kernel_size=5, strides=3, dilation_rate=1, activation='relu')(x)

    x = Conv1D(filters=38, kernel_size=5, strides=1, dilation_rate=3, activation='relu')(x)
    x = AveragePooling1D(pool_size=3)(x)

    x = BatchNormalization()(x)

    x = Dropout(rate=0.06957722835359469)(x)

    x = GaussianNoise(stddev=0.01751778593490914)(x)

    x = Conv1D(filters=21, kernel_size=1, activation='relu')(x)

    x1 = AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
    x1 = Conv1D(filters=79, kernel_size=1, padding='same', activation='relu')(x1)
    x2 = Conv1D(filters=79, kernel_size=1, padding='same', activation='relu')(x)
    x3 = Conv1D(filters=33, kernel_size=1, padding='same', activation='relu')(x)
    x3 = Conv1D(filters=79, kernel_size=3, padding='same', activation='relu')(x3)
    x4 = Conv1D(filters=33, kernel_size=1, padding='same', activation='relu')(x)
    x4 = Conv1D(filters=79, kernel_size=5, padding='same', activation='relu')(x4)
    x = concatenate([x1, x2, x3, x4], axis=2)
    x = AveragePooling1D(pool_size=2)(x)

    x = Dropout(rate=0.01512546661833562)(x)

    x2 = Conv1D(filters=177, kernel_size=1, padding='same', activation='relu')(x)
    x3 = Conv1D(filters=90, kernel_size=1, padding='same', activation='relu')(x)
    x3 = Conv1D(filters=177, kernel_size=3, padding='same', activation='relu')(x3)
    x4 = Conv1D(filters=90, kernel_size=1, padding='same', activation='relu')(x)
    x4 = Conv1D(filters=177, kernel_size=5, padding='same', activation='relu')(x4)
    x = concatenate([x2, x3, x4], axis=2)
    x = MaxPooling1D(pool_size=3)(x)

    x = BatchNormalization()(x)

    x = Dropout(rate=0.09102278901817017)(x)

    x = GaussianNoise(stddev=0.013956182085646474)(x)

    x = Conv1D(filters=class_count, kernel_size=1, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Softmax()(x)

    return x


def random_317(inputs, class_count):
    x = inputs

    x = GaussianNoise(stddev=0.0031280141265830064)(x)

    x = Conv1D(filters=13, kernel_size=3, strides=1, dilation_rate=1, activation='relu')(x)
    x = Conv1D(filters=13, kernel_size=3, strides=3, dilation_rate=1, activation='relu')(x)

    x = Dropout(rate=0.11511550791393513)(x)

    x = Conv1D(filters=22, kernel_size=3, strides=1, dilation_rate=1, activation='relu')(x)
    x = Conv1D(filters=22, kernel_size=7, strides=1, dilation_rate=1, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(rate=0.08801138466513796)(x)

    x = GaussianNoise(stddev=0.01261906780937131)(x)

    x = Conv1D(filters=34, kernel_size=3, strides=1, dilation_rate=1, activation='relu')(x)
    x = Conv1D(filters=34, kernel_size=3, strides=3, dilation_rate=1, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(rate=0.12827084537195393)(x)

    x = Conv1D(filters=20, kernel_size=1, activation='relu')(x)

    x1 = AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
    x1 = Conv1D(filters=53, kernel_size=1, padding='same', activation='relu')(x1)
    x2 = Conv1D(filters=53, kernel_size=1, padding='same', activation='relu')(x)
    x3 = Conv1D(filters=27, kernel_size=1, padding='same', activation='relu')(x)
    x3 = Conv1D(filters=53, kernel_size=3, padding='same', activation='relu')(x3)
    x4 = Conv1D(filters=27, kernel_size=1, padding='same', activation='relu')(x)
    x4 = Conv1D(filters=53, kernel_size=5, padding='same', activation='relu')(x4)
    x = concatenate([x1, x2, x3, x4], axis=2)
    x = MaxPooling1D(pool_size=3)(x)

    x = BatchNormalization()(x)

    x = Dropout(rate=0.07906722975035585)(x)

    x = GaussianNoise(stddev=0.03146882316978832)(x)

    x = Conv1D(filters=class_count, kernel_size=1, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Softmax()(x)

    return x


def random_268(inputs, class_count):
    x = inputs

    x = AveragePooling1D(pool_size=3, strides=2)(x)

    x1 = AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
    x1 = Conv1D(filters=12, kernel_size=1, padding='same', activation='relu')(x1)
    x3 = Conv1D(filters=8, kernel_size=1, padding='same', activation='relu')(x)
    x3 = Conv1D(filters=12, kernel_size=3, padding='same', activation='relu')(x3)
    x = concatenate([x1, x3], axis=2)
    x = MaxPooling1D(pool_size=2)(x)

    x = BatchNormalization()(x)

    x = Dropout(rate=0.0813733601425684)(x)

    x1 = AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
    x1 = Conv1D(filters=16, kernel_size=1, padding='same', activation='relu')(x1)
    x2 = Conv1D(filters=16, kernel_size=1, padding='same', activation='relu')(x)
    x3 = Conv1D(filters=5, kernel_size=1, padding='same', activation='relu')(x)
    x3 = Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')(x3)
    x4 = Conv1D(filters=5, kernel_size=1, padding='same', activation='relu')(x)
    x4 = Conv1D(filters=16, kernel_size=5, padding='same', activation='relu')(x4)
    x = concatenate([x1, x2, x3, x4], axis=2)
    x = MaxPooling1D(pool_size=2)(x)

    x = BatchNormalization()(x)

    x = Dropout(rate=0.003660480931626786)(x)

    x = GaussianNoise(stddev=0.03471230181839424)(x)

    x1 = AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
    x1 = Conv1D(filters=23, kernel_size=1, padding='same', activation='relu')(x1)
    x2 = Conv1D(filters=23, kernel_size=1, padding='same', activation='relu')(x)
    x3 = Conv1D(filters=13, kernel_size=1, padding='same', activation='relu')(x)
    x3 = Conv1D(filters=23, kernel_size=3, padding='same', activation='relu')(x3)
    x4 = Conv1D(filters=13, kernel_size=1, padding='same', activation='relu')(x)
    x4 = Conv1D(filters=23, kernel_size=5, padding='same', activation='relu')(x4)
    x = concatenate([x1, x2, x3, x4], axis=2)
    x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(rate=0.10725039245145246)(x)

    x = GaussianNoise(stddev=0.014715698696223467)(x)

    x = Conv1D(filters=35, kernel_size=5, strides=1, dilation_rate=2, activation='relu')(x)
    x = Conv1D(filters=35, kernel_size=3, strides=1, dilation_rate=1, activation='relu')(x)
    x = Conv1D(filters=35, kernel_size=5, strides=1, dilation_rate=1, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(rate=0.11214308791797134)(x)

    x = Conv1D(filters=7, kernel_size=1, activation='relu')(x)

    x = Conv1D(filters=65, kernel_size=3, strides=2, dilation_rate=1, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=39, kernel_size=1, activation='relu')(x)

    x = Conv1D(filters=101, kernel_size=3, strides=2, dilation_rate=1, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(rate=0.10503228953536241)(x)

    x = Conv1D(filters=class_count, kernel_size=1, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Softmax()(x)

    return x


def random_381(inputs, class_count):
    x = inputs

    x = GaussianNoise(stddev=0.034004904336028596)(x)

    x = Conv1D(filters=32, kernel_size=7, strides=1, dilation_rate=3, activation='relu')(x)
    x = Conv1D(filters=32, kernel_size=9, strides=1, dilation_rate=2, activation='relu')(x)
    x = Conv1D(filters=32, kernel_size=4, strides=2, dilation_rate=1, activation='relu')(x)

    x = Dropout(rate=0.046497062241251906)(x)

    x = Conv1D(filters=54, kernel_size=4, strides=1, dilation_rate=3, activation='relu')(x)
    x = Conv1D(filters=54, kernel_size=3, strides=1, dilation_rate=1, activation='relu')(x)
    x = MaxPooling1D(pool_size=3)(x)

    x = Conv1D(filters=90, kernel_size=3, strides=2, dilation_rate=1, activation='relu')(x)

    x = Dropout(rate=0.12339504634973077)(x)

    x = GaussianNoise(stddev=0.006245761590158084)(x)

    x = Conv1D(filters=120, kernel_size=3, strides=1, dilation_rate=1, activation='relu')(x)
    x = Conv1D(filters=120, kernel_size=5, strides=1, dilation_rate=1, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(rate=0.09818212478538961)(x)

    x = GaussianNoise(stddev=0.03623510126409917)(x)

    x = Conv1D(filters=65, kernel_size=1, activation='relu')(x)

    x = Conv1D(filters=class_count, kernel_size=1, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Softmax()(x)

    return x


def random_217(inputs, class_count):
    x = inputs

    x = GaussianNoise(stddev=0.02800324054815674)(x)

    x = AveragePooling1D(pool_size=2, strides=2)(x)

    x = Conv1D(filters=23, kernel_size=3, strides=1, dilation_rate=1, activation='relu')(x)
    x = Conv1D(filters=23, kernel_size=5, strides=1, dilation_rate=1, activation='relu')(x)
    x = Conv1D(filters=23, kernel_size=3, strides=3, dilation_rate=1, activation='relu')(x)

    x = Dropout(rate=0.12431498467028188)(x)

    x = GaussianNoise(stddev=0.020629561240189956)(x)

    x = Conv1D(filters=30, kernel_size=6, strides=1, dilation_rate=1, activation='relu')(x)
    x = Conv1D(filters=30, kernel_size=3, strides=1, dilation_rate=1, activation='relu')(x)
    x = Conv1D(filters=30, kernel_size=3, strides=1, dilation_rate=2, activation='relu')(x)
    x = Conv1D(filters=30, kernel_size=8, strides=1, dilation_rate=1, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(rate=0.13875930485777344)(x)

    x = GaussianNoise(stddev=0.010444263294661789)(x)

    x = Conv1D(filters=23, kernel_size=1, activation='relu')(x)

    x2 = Conv1D(filters=51, kernel_size=1, padding='same', activation='relu')(x)
    x3 = Conv1D(filters=11, kernel_size=1, padding='same', activation='relu')(x)
    x3 = Conv1D(filters=51, kernel_size=3, padding='same', activation='relu')(x3)
    x4 = Conv1D(filters=11, kernel_size=1, padding='same', activation='relu')(x)
    x4 = Conv1D(filters=51, kernel_size=5, padding='same', activation='relu')(x4)
    x = concatenate([x2, x3, x4], axis=2)
    x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(rate=0.11156457553830097)(x)

    x = Conv1D(filters=118, kernel_size=3, strides=1, dilation_rate=3, activation='relu')(x)
    x = Conv1D(filters=118, kernel_size=3, strides=3, dilation_rate=1, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=19, kernel_size=1, activation='relu')(x)

    x = Flatten()(x)

    x = Dense(51, activation='relu')(x)
    x = Dense(62, activation='relu')(x)
    x = Dense(213, activation='relu')(x)

    x = Dropout(rate=0.1564586327010344)(x)

    x = Dense(class_count, activation='softmax')(x)

    return x


def random_080_with_gap(inputs, class_count):
    x = inputs

    x = GaussianNoise(stddev=0.01)(x)

    x = Conv1D(filters=42, kernel_size=3, strides=2, activation='relu')(x)

    x = Dropout(rate=0.15)(x)

    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(rate=0.15)(x)

    x = Conv1D(filters=14, kernel_size=1, activation='relu')(x)

    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.15)(x)

    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.15)(x)

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
    x = Dropout(rate=0.15)(x)

    x = Conv1D(filters=42, kernel_size=3, strides=2, activation='relu')(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.15)(x)

    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.15)(x)

    x = Conv1D(filters=class_count, kernel_size=1, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Softmax()(x)

    return x


def random_080_with_gap_more_norm(inputs, class_count):
    x = inputs

    x = GaussianNoise(stddev=0.01)(x)

    x = Conv1D(filters=42, kernel_size=3, strides=2, activation='relu')(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.15)(x)

    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.15)(x)

    x = Conv1D(filters=14, kernel_size=1, activation='relu')(x)

    # Maybe add normalisation and dropout here too?

    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.15)(x)

    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.15)(x)

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
    x = Dropout(rate=0.15)(x)

    x = Conv1D(filters=42, kernel_size=3, strides=2, activation='relu')(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.15)(x)

    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.15)(x)

    x = Conv1D(filters=class_count, kernel_size=1, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Softmax()(x)

    return x


def random_080_norm_and_drop_after_bottleneck(inputs, class_count):
    x = inputs

    x = GaussianNoise(stddev=0.01)(x)

    x = Conv1D(filters=42, kernel_size=3, strides=2, activation='relu')(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.15)(x)

    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.15)(x)

    x = Conv1D(filters=14, kernel_size=1, activation='relu')(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.15)(x)

    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.15)(x)

    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.15)(x)

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
    x = Dropout(rate=0.15)(x)

    x = Conv1D(filters=42, kernel_size=3, strides=2, activation='relu')(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.15)(x)

    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=42, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.15)(x)

    x = Conv1D(filters=class_count, kernel_size=1, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Softmax()(x)

    return x
