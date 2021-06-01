import os

from tensorflow.python.keras.applications.xception import preprocess_input
from tensorflow.keras import layers

from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.keras.layers.experimental import preprocessing



TF_WEIGHTS_PATH = (
    'https://storage.googleapis.com/tensorflow/keras-applications/'
    'xception/xception_weights_tf_dim_ordering_tf_kernels.h5')
TF_WEIGHTS_PATH_NO_TOP = (
    'https://storage.googleapis.com/tensorflow/keras-applications/'
    'xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')

layers = VersionAwareLayers()


def Xception(input_shape=None, classes=1000):

    include_top = False
    weights = 'imagenet'
    input_tensor = None
    input_shape = input_shape
    pooling = 'avg'
    classifier_activation = 'softmax'

    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=71,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    # augments
    # x = preprocessing.RandomFlip('horizontal_and_vertical')(img_input)
    # x = preprocessing.RandomRotation(0.2)(x)
    # x = preprocessing.RandomZoom(0.2)(x)
    ########################################################################
    # x = layers.Lambda(preprocess_input)(x)   # added by me
    x = layers.Conv2D(
        32, (3, 3),
        strides=(2, 2),
        use_bias=False,
        name='block1_conv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
    x = layers.Activation('relu', name='block1_conv1_act')(x)
    x = layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
    x = layers.Activation('relu', name='block1_conv2_act')(x)

    residual = layers.Conv2D(
        128, (1, 1), strides=(2, 2), padding='same', use_bias=False, name='res1_conv')(x)
    residual = layers.BatchNormalization(axis=channel_axis, name='res1_bn')(residual)

    x = layers.SeparableConv2D(
        128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block2_sepconv2_act')(x)
    x = layers.SeparableConv2D(
        128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(
        256, (1, 1), strides=(2, 2), padding='same', use_bias=False, name='res2_conv')(x)
    residual = layers.BatchNormalization(axis=channel_axis, name='res2_bn')(residual)

    x = layers.Activation('relu', name='block3_sepconv1_act')(x)
    x = layers.SeparableConv2D(
        256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block3_sepconv2_act')(x)
    x = layers.SeparableConv2D(
        256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(
        728, (1, 1), strides=(2, 2), padding='same', use_bias=False, name='res3_conv')(x)
    residual = layers.BatchNormalization(axis=channel_axis, name='res3_bn')(residual)

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)
    x = layers.SeparableConv2D(
        728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block4_sepconv2_act')(x)
    x = layers.SeparableConv2D(
        728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block4_pool')(x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = layers.SeparableConv2D(
            728, (3, 3),
            padding='same',
            use_bias=False,
            name=prefix + '_sepconv1')(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=prefix + '_sepconv1_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = layers.SeparableConv2D(
            728, (3, 3),
            padding='same',
            use_bias=False,
            name=prefix + '_sepconv2')(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=prefix + '_sepconv2_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = layers.SeparableConv2D(
            728, (3, 3),
            padding='same',
            use_bias=False,
            name=prefix + '_sepconv3')(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])

    residual = layers.Conv2D(
        1024, (1, 1), strides=(2, 2), padding='same', use_bias=False, name='res4_conv')(x)
    residual = layers.BatchNormalization(axis=channel_axis, name='res4_bn')(residual)

    x = layers.Activation('relu', name='block13_sepconv1_act')(x)
    x = layers.SeparableConv2D(
        728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='block13_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block13_sepconv2_act')(x)
    x = layers.SeparableConv2D(
        1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='block13_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block13_pool')(x)
    x = layers.add([x, residual])

    x = layers.SeparableConv2D(
        1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='block14_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv1_act')(x)

    x = layers.SeparableConv2D(
        2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='block14_sepconv2_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv2_act')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(classes, activation=classifier_activation,
                         name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # customize last layers
    x = layers.Dropout(0.5, name='dropout')(x)
    x = layers.Dense(classes, activation='softmax', name='dense')(x)
    ###############################################################

    # Create model.
    model = training.Model(inputs, x, name='xception')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = data_utils.get_file(
                'xception_weights_tf_dim_ordering_tf_kernels.h5',
                TF_WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='0a58e3b7378bc2990ea3b43d5981f1f6')
        else:
            weights_path = os.path.join(os.path.dirname(__file__), 'xception_weights_notop.hdf5')
        model.load_weights(weights_path, by_name=True)
    elif weights is not None:
        model.load_weights(weights)

    return model


if __name__ == '__main__':
    model = Xception(input_shape=(299,299,3), classes=12)
    # model.save_weights(os.path.join('models','xception_weights_notop.hdf5'))
    model.summary()
