from tensorflow.python.keras.applications.mobilenet import preprocess_input, MobileNet
from tensorflow.keras import layers, Model


def MobileNet0(input_shape=None, classes=1000):

    model = MobileNet(include_top=False,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=input_shape,
                     pooling='avg',
                     classes=classes,
                     classifier_activation='softmax')
    # for layer in model.layers[:-7]:
    #     layer.trainable = False
    x = layers.Lambda(preprocess_input)(model.input)
    x = model(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(classes, activation='softmax')(x)
    model = Model(model.input, x)


    return model

if __name__ == '__main__':
    model = MobileNet0(input_shape=(199, 199, 3), classes=12)
    # model = MobileNet(include_top=False,
    #                  weights='imagenet',
    #                  input_tensor=None,
    #                  input_shape=(199,199,3),
    #                  pooling='avg',
    #                  classes=12,
    #                  classifier_activation='softmax')
    model.summary()
