# Keras imports
import keras.backend as K
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.engine.topology import merge

# Paper:


def build_densenetFCN(in_shape=(3, 224, 224), n_classes=1000, weight_decay=0.,
                      freeze_layers_from='base_model', path_weights=None):

    # Layers before dense blocks
    inp = Input(shape=in_shape)
    n_filter = 32
    x = Convolution2D(n_filter, 7, 7, subsample=(2, 2),
                      border_mode='same', name='conv1')(inp)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Add dense blocks: 4 blocks of 10 layers -> 40 layers
    growth_rate = 12
    n_layers = 10
    dropout_fraction = 0.0
    x, n_filter = denseBlock(x, n_layers, growth_rate, n_filter, dropout_fraction)
    x = transitionLayer(x, n_filter)
    x, n_filter = denseBlock(x, n_layers, growth_rate, n_filter, dropout_fraction)
    x = transitionLayer(x, n_filter)
    x, n_filter = denseBlock(x, n_layers, growth_rate, n_filter, dropout_fraction)
    x = transitionLayer(x, n_filter)
    x, n_filter = denseBlock(x, n_layers, growth_rate, n_filter, dropout_fraction)

    # Add final layers
    x = GlobalAveragePooling2D(dim_ordering=K.image_dim_ordering())(x)  # Base model
    predictions = Dense(n_classes, activation='softmax',
                        name='classification_layer')(x)

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


def denseBlock(x, n_layers, growth_rate, n_filter, dropout_fraction):
    past_features = [x]

    if K.image_dim_ordering() == 'th':
        concat_axis = 1
    elif K.image_dim_ordering() == 'tf':
        concat_axis = -1

    for i in range(n_layers):
        x = BatchNormalization(mode=0, axis=concat_axis)(x)
        x = Activation('relu')(x)
        x = Convolution2D(growth_rate, 1, 1, border_mode='same')(x)
        x = BatchNormalization(mode=0, axis=concat_axis)(x)
        x = Activation('relu')(x)
        x = Convolution2D(growth_rate, 3, 3, border_mode='same')(x)
        if dropout_fraction != 0:
            x = Dropout(dropout_fraction)(x)

        past_features.append(x)
        x = merge(past_features, mode='concat', concat_axis=concat_axis)
        n_filter += growth_rate

    output = x

    return output, n_filter


def transitionLayer(x, n_filter):
    x = Convolution2D(n_filter, 1, 1, border_mode="same")(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    output = x
    return output
