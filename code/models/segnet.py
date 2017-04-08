import keras.backend as K
from keras.models import Model
from keras.layers import Input, UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from layers.ourlayers import DePool2D, NdSoftmax, CropLayer2D

# Paper: http://arxiv.org/abs/1505.07293


def build_segnet(in_shape=(3, 224, 224), n_classes=11, weight_decay=0.,
                 freeze_layers_from='base_model', basic=False):

    # TODO: weight decay

    kernel = 3  # kernel size
    l2r = 0.  # L2 regularization

    if K.image_dim_ordering() == 'th':
        concat_axis = 1
    elif K.image_dim_ordering() == 'tf':
        concat_axis = -1

    inp = Input(shape=in_shape)

    if basic:
        predictions = segnet_basic(inp, 7, concat_axis, n_classes, l2r)
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

    x = Convolution2D(128, kernel, kernel, border_mode='same')(pool1)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = Convolution2D(128, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Activation('relu')(x)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Convolution2D(256, kernel, kernel, border_mode='same')(pool2)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = Convolution2D(256, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = Convolution2D(256, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Activation('relu')(x)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Convolution2D(512, kernel, kernel, border_mode='same')(pool3)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = Convolution2D(512, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = Convolution2D(512, kernel, kernel, border_mode='same')(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    x = Activation('relu')(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Convolution2D(512, kernel, kernel, border_mode='same')(pool4)
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


def segnet_basic(inp, kernel, concat_axis, n_classes, l2r):

    # Encoding layers
    enc1 = downsampling_block_basic(inp, 64, kernel, concat_axis, l2(l2r))
    enc2 = downsampling_block_basic(enc1, 64, kernel, concat_axis, l2(l2r))
    enc3 = downsampling_block_basic(enc2, 64, kernel, concat_axis, l2(l2r))
    enc4 = downsampling_block_basic(enc3, 64, kernel, concat_axis, l2(l2r))

    # Decoding layers
    dec1 = upsampling_block_basic(enc4, 64, kernel, enc4, concat_axis, l2(l2r))
    dec2 = upsampling_block_basic(dec1, 64, kernel, enc3, concat_axis, l2(l2r))
    dec3 = upsampling_block_basic(dec2, 64, kernel, enc2, concat_axis, l2(l2r))
    dec4 = upsampling_block_basic(dec3, 64, kernel, enc1, concat_axis, l2(l2r))

    l1 = Convolution2D(n_classes, 1, 1, border_mode='valid')(dec4)
    score = CropLayer2D(inp, name='score')(l1)
    predictions = NdSoftmax()(score)

    return predictions


def downsampling_block_basic(inputs, n_filters, kernel, concat_axis=-1,
                             W_regularizer=None):
    # This extra padding is used to prevent problems with different input
    # sizes. At the end the crop layer remove extra paddings
    pad = ZeroPadding2D(padding=(1, 1))(inputs)
    conv = Convolution2D(n_filters, kernel, kernel, border_mode='same',
                         W_regularizer=W_regularizer)(pad)
    bn = BatchNormalization(mode=0, axis=concat_axis)(conv)
    act = Activation('relu')(bn)
    output = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act)

    return output


def upsampling_block_basic(inputs, n_filters, kernel, unpool_layer=None,
                           concat_axis=-1, W_regularizer=None,
                           use_unpool=True):
    if use_unpool:
        up = DePool2D(unpool_layer)(inputs)
        return up
    else:
        up = UpSampling2D()(inputs)
        conv = Convolution2D(n_filters, kernel, kernel, border_mode='same',
                             W_regularizer=W_regularizer)(up)
        bn = BatchNormalization(mode=0, axis=concat_axis,)(conv)
        return bn
