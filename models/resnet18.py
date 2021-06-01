from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2


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
    filters1, filters2 = filters
    bn_axis = 3     # if channels_first, it wiil be 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, kernel_size,
                      padding='same',
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
    # x = layers.Activation('relu')(x)


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
    filters1, filters2 = filters
    bn_axis = 3      # if channels_first, it wiil be 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, kernel_size, strides=strides,
                      padding='same',
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

    shortcut = layers.Conv2D(filters2, kernel_size, strides=strides,
                             padding='same',
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(weight_decay),
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def ResNet18(input_shape=None,
             classes=29):
    bn_axis = 3
    weight_decay = 1e-4
    input_tensor = layers.Input(input_shape, name='resnet18_input')


    x = layers.Conv2D(64, (7, 7),
                      padding='same',
                      kernel_initializer='he_normal',
                      name='resnet18_conv1',
                      kernel_regularizer=l2(weight_decay),
                      strides=(2, 2))(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = identity_block(x, (3,3), [48,64], stage=2, block='1b')
    x = identity_block(x, (3,3), [48,64], stage=3, block='1c')

    x = conv_block(x, (3,3), [96,128], stage=3, block='2a', strides=(2,2))
    x = identity_block(x, (3,3), [96,128], stage=3, block='2b')

    x = conv_block(x, (3,3), [128,256], stage=4, block='3a', strides=(2,2))
    x = identity_block(x, (3,3), [128,256], stage=4, block='3b')

    x = conv_block(x, (3,3), [256,512], stage=5, block='4a', strides=(2,2))
    x = identity_block(x, (3,3), [256,512], stage=5, block='4b')

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(classes, activation='softmax', kernel_regularizer=l2(weight_decay), name='fc1000')(x)

    model = Model(input_tensor, x, name='resnet18_model')
    return model

def ResNet34(input_shape=None,
             classes=29):
    bn_axis = 3
    weight_decay = 1e-4
    input_tensor = layers.Input(input_shape, name='resnet34_input')


    x = layers.Conv2D(64, (7, 7),
                      padding='same',
                      kernel_initializer='he_normal',
                      name='resnet34_conv1',
                      kernel_regularizer=l2(weight_decay),
                      strides=(2, 2))(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((6, 3), strides=(2, 2))(x)

    x = identity_block(x, (3,3), [48,64], stage=2, block='1b')
    x = identity_block(x, (3,3), [48,64], stage=3, block='1c')
    x = identity_block(x, (3,3), [48,64], stage=3, block='1d')

    x = conv_block(x, (3,3), [96,128], stage=3, block='2a', strides=(2,2))
    x = identity_block(x, (3,3), [96,128], stage=3, block='2b')
    x = identity_block(x, (3,3), [96,128], stage=3, block='2c')
    x = identity_block(x, (3,3), [96,128], stage=3, block='2d')

    x = conv_block(x, (3,3), [128,256], stage=4, block='3a', strides=(2,2))
    x = identity_block(x, (3,3), [128,256], stage=4, block='3b')
    x = identity_block(x, (3,3), [128,256], stage=4, block='3c')
    x = identity_block(x, (3,3), [128,256], stage=4, block='3d')
    x = identity_block(x, (3,3), [128,256], stage=4, block='3e')
    x = identity_block(x, (3,3), [128,256], stage=4, block='3f')

    x = conv_block(x, (3,3), [256,512], stage=5, block='4a', strides=(2,2))
    x = identity_block(x, (3,3), [256,512], stage=5, block='4b')
    x = identity_block(x, (3,3), [256,512], stage=5, block='4c')


    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(classes, activation='softmax', kernel_regularizer=l2(weight_decay), name='fc1000')(x)

    model = Model(input_tensor, x, name='resnet34_model')
    return model

if __name__ == '__main__':
    # model = ResNet50([1866, 4088, 3], classes=100)
    model = ResNet18(input_shape=[4096, 3000, 3], classes=29)
    model.summary()
    model.save('models/resnet18.h5')
