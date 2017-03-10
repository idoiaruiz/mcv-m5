# Keras imports
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input,
AveragePooling2D, GlobalAveragePooling2D
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)

# Paper:


def build_denseNet(img_shape=(3, 224, 224), n_classes=1000, freeze_layers_from='base_model'):

    # Layers before dense blocks
    inp = Input(shape=img_shape)
    x = Convolution2D(32, 7, 7, subsample=(2, 2),
                      border_mode='same', name='conv1')(inp)
    x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Add dense blocks


    # Add final layers
    x = GlobalAveragePooling2D(dim_ordering=K.image_dim_ordering())(x)
    x = Flatten(name="flatten")(x)
    predictions = Dense(n_classes, activation='softmax', name='classification_layer')(x)

    # This is the model we will train
    model = Model(input=inp, output=predictions)

    # Freeze some layers
    if freeze_layers_from is not None:
        if freeze_layers_from == 'base_model':
            print ('   Freezing base model layers')
            for layer in model.layers:
                layer.trainable = False
        else:
            for i, layer in enumerate(model.layers):
                print(i, layer.name)
            print ('   Freezing from layer 0 to ' + str(freeze_layers_from))
            for layer in model.layers[:freeze_layers_from]:
                layer.trainable = False
            for layer in model.layers[freeze_layers_from:]:
                layer.trainable = True

    return model


def denseBlock(x, k, nb_filter):
    past_features = x

    if K.image_dim_ordering() == 'th':
        concat_axis = 1
    elif K.image_dim_ordering() == 'tf':
        concat_axis = -1

    for ki in range(k):
        x = BatchNormalization(mode=0, axis=1)(x)
        x = Activation('relu')(x)
        x = Convolution2D(k, 1, 1, border_mode='same')(x)
        x = BatchNormalization(mode=0, axis=1)(x)
        x = Activation('relu')(x)
        x = Convolution2D(k, 3, 3, border_mode='same')(x)
        #x = Dropout(dropout_fraction)(x)

        past_features.append(x)
        x = merge(past_features, mode='concat', concat_axis=concat_axis)
        nb_filter += k

    output = x, nb_filter

    return output


def transitionLayer(x, nb_filter):
    x = Convolution2D(nb_filter, 1, 1, border_mode="same")(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    output = x
    return output
