from tensorflow.keras.applications.efficientnet import *
from tensorflow.keras import Model, layers


def Efficient4(input_shape=None, classes=1000):
    """
    default input shape is (380, 380, 3)
    """
    model = EfficientNetB4(include_top=False,
                            weights='imagenet',
                            input_tensor=None,
                            input_shape=input_shape,
                            pooling='avg',
                            classes=1000,
                            classifier_activation='softmax')
    x = layers.Lambda(preprocess_input)(model.input)
    x = model(x)
    # x = layers.Dropout(0.5)(x)
    x = layers.Dense(classes)(x)

    model = Model(model.input, x)

    return model


if __name__ == '__main__':
    model = Efficient4((1024,1024,3),3)
    model.summary()