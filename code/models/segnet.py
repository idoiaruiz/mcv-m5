import keras.backend as K
from keras.models import Model
from keras.layers.core import Activation, Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

from layers.ourlayers import DePool2D, NdSoftmax, CropLayer2D

# Paper: http://arxiv.org/abs/1505.07293


def build_segnet(in_shape=(3, 224, 224), n_classes=11, weight_decay=0.,
                 freeze_layers_from='base_model', basic=False):

    # TODO: weight decay

    kernel = 3  # kernel size

    if K.image_dim_ordering() == 'th':
        concat_axis = 1
    elif K.image_dim_ordering() == 'tf':
        concat_axis = -1

    inp = Input(shape=in_shape)

    if basic:
        predictions = segnet_basic(inp, kernel, concat_axis, n_classes)
    else:
        predictions = segnet_VGG(inp, kernel, concat_axis, n_classes)

    model = Model(input=inp, output=predictions)

    # Freeze some layers
    if freeze_layers_from is not None:
        if freeze_layers_from == 'base_model':
            raise ValueError('Please enter the layer id, instead of "base_model"'
                             ' for the "freeze_layers_from" config parameter')
        else:
            for i, layer in enumerate(model.layers):
                print(i, layer.name)
            print ('   Freezing from layer 0 to ' + str(freeze_layers_from))
            for layer in model.layers[:freeze_layers_from]:
                layer.trainable = False
            for layer in model.layers[freeze_layers_from:]:
                layer.trainable = True

    return model


def segnet_VGG(inp, kernel, concat_axis, n_classes):
    # Encoding layers: VGG 13 convolutional layers
    x = Convolution2D(64, kernel, kernel, border_mode='same')(inp)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = Convolution2D(64, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Activation('relu')(x)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Convolution2D(128, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = Convolution2D(128, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Activation('relu')(x)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Convolution2D(256, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = Convolution2D(256, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = Convolution2D(256, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Activation('relu')(x)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Convolution2D(512, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = Convolution2D(512, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = Convolution2D(512, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Activation('relu')(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Convolution2D(512, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = Convolution2D(512, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = Convolution2D(512, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Activation('relu')(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Decoding layers
    x = DePool2D(pool2d_layer=pool5, size=(2, 2))(pool5)
    x = Convolution2D(512, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Convolution2D(512, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Convolution2D(512, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)

    x = DePool2D(pool2d_layer=pool4, size=(2, 2))(x)
    x = Convolution2D(512, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Convolution2D(512, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Convolution2D(256, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)

    x = DePool2D(pool2d_layer=pool3, size=(2, 2))(x)
    x = Convolution2D(256, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Convolution2D(256, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Convolution2D(128, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)

    x = DePool2D(pool2d_layer=pool2, size=(2, 2))(x)
    x = Convolution2D(128, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Convolution2D(64, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)

    x = DePool2D(pool2d_layer=pool1, size=(2, 2))(x)
    x = Convolution2D(64, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Convolution2D(64, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)

    x = Convolution2D(n_classes, 1, 1, border_mode='valid')(x)

    score = CropLayer2D(inp, name='score')(x)
    predictions = NdSoftmax()(score)

    return predictions


def segnet_basic(inp, kernel, concat_axis, n_classes):
    # TODO: implement

    return None
