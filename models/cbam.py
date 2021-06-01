# Author  : WangXiao
# File    : cabm.py
# Function: 搭建cbam模块
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
from tensorflow.keras.layers import Layer,GlobalAveragePooling2D,GlobalMaxPooling2D,Reshape,Conv2D,multiply,Concatenate
from tensorflow.python.keras import backend as K
from tensorflow.keras import Sequential



def channel_attention(input_feature,rotio):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel_num = input_feature.shape[channel_axis]

    avg_out = GlobalAveragePooling2D()(input_feature)
    max_out = GlobalMaxPooling2D()(input_feature)

    avg_out = Reshape((1,1,channel_num))(avg_out)
    max_out = Reshape((1,1,channel_num))(max_out)


    shared_layer = Sequential(
        [Conv2D(channel_num//rotio,kernel_size=1,strides=1,activation='relu',use_bias=False),
        Conv2D(channel_num,kernel_size=1,strides=1,use_bias=False)]
                              )

    avg_out = shared_layer(avg_out)

    max_out = shared_layer(max_out)

    y = tf.keras.activations.sigmoid(avg_out+max_out)
    y = multiply([input_feature,y])
    return y


class channel_min(Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs, **kwargs):
        if K.image_data_format() == 'channels_first':
            out_put = K.min(inputs,axis=1,keepdims=True)
        else:
            out_put = K.min(inputs,axis=-1,keepdims=True)
        return out_put


class channel_max(Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs, **kwargs):
        if K.image_data_format() == 'channels_first':
            out_put = K.max(inputs, axis=1, keepdims=True)
        else:
            out_put = K.max(inputs, axis=-1, keepdims=True)
        return out_put


def spatial_attention(input_feature,kernel_size=7):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    y_mean = channel_min()(input_feature)
    y_max = channel_max()(input_feature)
    y = Concatenate(axis=channel_axis)([y_mean,y_max])
    y = Conv2D(filters=1,kernel_size=kernel_size,activation='sigmoid',strides=1,padding='same',use_bias=False)(y)

    return multiply([input_feature,y])


def cbam_block(input_feature,rotio,kernel_size=7):
    y = channel_attention(input_feature,rotio)
    y = spatial_attention(y,kernel_size)
    return y



if __name__ == '__main__':
    inputs = tf.keras.layers.Input(shape=(224,224,60))
    out = cbam_block(inputs,6)
    model = tf.keras.Model(inputs,out)
    model.summary()
    model.save('models/cbam.h5')