from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import imagenet_utils


def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(x, data_format=data_format, mode='caffe')


def identity_block(input_tensor, kernel_size, filters, stage, block, weight_decay=5e-4):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3  # if channels_first, it wiil be 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay),
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay),
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay),
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               weight_decay=5e-4):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3  # if channels_first, it wiil be 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay),
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay),
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay),
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(weight_decay),
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def ResNet50(input_shape=None,
             classes=29):
    bn_axis = 3
    weight_decay = 1e-4
    input_tensor = layers.Input(input_shape, name='resnet50_input')

    x = layers.Conv2D(64, (12, 6),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='resnet50_conv1',
                      kernel_regularizer=l2(weight_decay),
                      strides=(2, 2))(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((6, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(classes, activation='softmax', kernel_regularizer=l2(weight_decay), name='fc1000')(x)

    model = Model(input_tensor, x, name='resnet50_model')

    return model


def ResNet110(input_shape=None,
              classes=29):
    bn_axis = 3
    weight_decay = 1e-4
    input_tensor = layers.Input(input_shape, name='resnet110_input')

    x = layers.Lambda(preprocess_input)(input_tensor)
    x = layers.Conv2D(64, (12, 6),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='resnet50_conv1',
                      kernel_regularizer=l2(weight_decay),
                      strides=(2, 2))(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((6, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='g')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='h')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='i')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='j')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='k')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='l')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='m')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='n')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='o')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='p')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='q')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='r')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='s')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='t')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='u')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='v')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='w')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(classes, activation='softmax', kernel_regularizer=l2(weight_decay), name='fc1000')(x)

    model = Model(input_tensor, x, name='resnet110_model')

    return model


def ResNet152(input_shape=None,
              classes=29):
    bn_axis = 3
    weight_decay = 1e-4
    input_tensor = layers.Input(input_shape, name='resnet152_input')

    x = layers.Conv2D(64, (12, 6),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='resnet50_conv1',
                      kernel_regularizer=l2(weight_decay),
                      strides=(2, 2))(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((6, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='e')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='f')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='g')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='h')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='g')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='h')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='i')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='j')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='k')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='l')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='m')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='n')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='o')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='p')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='q')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='r')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='s')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='t')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='u')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='v')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='w')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='x')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='y')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='z')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='27')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='28')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='29')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='30')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='31')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='32')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='33')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='34')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='35')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='36')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(classes, activation='softmax', kernel_regularizer=l2(weight_decay), name='fc1000')(x)

    model = Model(input_tensor, x, name='resnet152_model')

    return model


def bin_model(input_shape=None,
              classes=29):
    bn_axis = 3
    weight_decay = 1e-4
    input_tensor = layers.Input(input_shape, name='bin_model_input')

    x = layers.Conv2D(64, (7, 7),
                      padding='same',
                      kernel_initializer='he_normal',
                      name='conv1',
                      kernel_regularizer=l2(weight_decay),
                      strides=(2, 2))(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = layers.Conv2D(64, (7, 7),
                      padding='same',
                      kernel_initializer='he_normal',
                      name='conv2',
                      kernel_regularizer=l2(weight_decay),
                      strides=(2, 2))(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv2')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = layers.Conv2D(64, (7, 7),
                      padding='same',
                      kernel_initializer='he_normal',
                      name='conv3',
                      kernel_regularizer=l2(weight_decay),
                      strides=(2, 2))(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv3')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3),
                      padding='same',
                      kernel_initializer='he_normal',
                      name='conv4',
                      kernel_regularizer=l2(weight_decay),
                      strides=(2, 2))(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv4')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3),
                      padding='same',
                      kernel_initializer='he_normal',
                      name='conv5',
                      kernel_regularizer=l2(weight_decay),
                      strides=(2, 2))(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv5')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3),
                      padding='same',
                      kernel_initializer='he_normal',
                      name='conv6',
                      kernel_regularizer=l2(weight_decay),
                      strides=(2, 2))(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv6')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3),
                      padding='same',
                      kernel_initializer='he_normal',
                      name='conv7',
                      kernel_regularizer=l2(weight_decay),
                      strides=(2, 2))(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv7')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(classes, activation='softmax', kernel_regularizer=l2(weight_decay), name='fc')(x)

    model = Model(input_tensor, x, name='bin_model')
    return model


if __name__ == '__main__':
    # model = ResNet50([1866, 4088, 3], classes=100)
    model = bin_model(input_shape=[4096, 3000, 3], classes=29)
    model.summary()
