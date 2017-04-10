# Keras imports
import keras.backend as K
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Model
from keras.layers import Dropout, Activation, Input
from keras.layers.normalization import BatchNormalization
from keras.engine.topology import merge
from layers.deconv import Deconvolution2D
from layers.ourlayers import  NdSoftmax
# Paper: https://arxiv.org/pdf/1611.09326.pdf


def build_densenet_segmentation(in_shape = (3, 224, 224), n_classes = 1000, weight_decay = 0.,
                   freeze_layers_from = 'base_model', path_weights = None):

    #####################
    # First Convolution #
    #####################
    
    inp = Input(shape = in_shape)
    n_filter = 48
    x = Convolution2D(n_filter, 3, 3, subsample = (1, 1),
                      border_mode = 'same')(inp)
    
    #####################
    # Downsampling path #
    #####################
    
    
    growth_rate = 16
    dropout_fraction = 0.2
    n_layers_down = [4, 5, 7, 10, 12]
    skip_connection_list = []
    for i in range(len(n_layers_down)):
        #Dense block
        x, n_filter = denseBlock_down(x, n_layers_down[i], growth_rate, n_filter, dropout_fraction)
        
        print('number of filters = ' + str(x._keras_shape[1]))
        
        x = transition_down_Layer(x, n_filter, dropout_fraction)
        # At the end of the dense block, the current output is stored in the skip_connections list
        skip_connection_list.append(x)
        
    skip_connection_list = skip_connection_list[::-1]

    #####################
    #     Bottleneck    #
    ##################### 
    n_layers = 15
    # We store now the output of the next dense block in a list(block_to_upsample). 
    # We will only upsample these new feature maps
    x, n_filter, block_to_upsample = denseBlock_up(x, n_layers, growth_rate, n_filter, dropout_fraction)
    print('number of filters = ' + str(x._keras_shape[1]))
    
    
    # Add dense blocks of upsampling path

    n_layers_up = [15, 12, 10, 7, 5, 4]
    for j in range(1, len(n_layers_up)):
    
        # Transition Up ( Upsampling + concatenation with the skip connection)
        n_filters_keep = growth_rate * n_layers_up[j - 1]
        print('number of filters to keep = ' + str(n_filters_keep))

        x = transition_up_Layer(skip_connection_list[j - 1], block_to_upsample, n_filters_keep)
        x, n_filter, block_to_upsample = denseBlock_up(x, n_layers_up[j], growth_rate, n_filter, dropout_fraction)
        print('number of filters = ' + str(x._keras_shape[1]))
        
    #Last convolution    
    x = Convolution2D(n_classes, 1, 1, subsample = (1, 1),
                      border_mode = 'same')(x)
    
    #####################
    #      Softmax      #
    #####################
    predictions = NdSoftmax()(x)

    # This is the model we will train
    model = Model(input = inp, output = predictions)
#    model.summary()
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


def denseBlock_down(x, n_layers, growth_rate, n_filter, dropout_fraction):
    past_features = [x]

    if K.image_dim_ordering() == 'th':
        concat_axis = 1
    elif K.image_dim_ordering() == 'tf':
        concat_axis = -1

    for i in range(n_layers):
        x = BatchNormalization(mode = 0, axis = concat_axis)(x)
        x = Activation('relu')(x)
        x = Convolution2D(growth_rate, 3, 3, border_mode = 'same')(x)
        if dropout_fraction != 0:
            x = Dropout(dropout_fraction)(x)
            
        

        past_features.append(x)
        x = merge(past_features, mode = 'concat', concat_axis = concat_axis)
        n_filter += growth_rate

    output = x

    return output, n_filter

def denseBlock_up(x, n_layers, growth_rate, n_filter, dropout_fraction):
    past_features = [x]
    # We store now the output of the next dense block in a list. 
    # We will only upsample these new feature maps
    block_to_upsample = []
    if K.image_dim_ordering() == 'th':
        concat_axis = 1
    elif K.image_dim_ordering() == 'tf':
        concat_axis = -1

    for i in range(n_layers):
        x = BatchNormalization(mode = 0, axis = concat_axis)(x)
        x = Activation('relu')(x)
        x = Convolution2D(growth_rate, 3, 3, border_mode = 'same')(x)
        if dropout_fraction != 0:
            x = Dropout(dropout_fraction)(x)
            
        
        block_to_upsample.append(x)
        past_features.append(x)
        x = merge(past_features, mode = 'concat', concat_axis = concat_axis)
        n_filter += growth_rate

    output = x
    

    return output, n_filter, block_to_upsample

def transition_down_Layer(x, n_filter, dropout_fraction):
    if K.image_dim_ordering() == 'th':
        concat_axis = 1
    elif K.image_dim_ordering() == 'tf':
        concat_axis = -1
        
    x = BatchNormalization(mode = 0, axis = concat_axis)(x) 
    x = Activation('relu')(x)
        
    x = Convolution2D(n_filter, 1, 1, border_mode = "same")(x)
    if dropout_fraction != 0:
        x = Dropout(dropout_fraction)(x)
    x = MaxPooling2D((2, 2), strides = (2, 2))(x)

    output = x
    return output

def transition_up_Layer(skip_connection, block_to_upsample, n_filters_keep):
    if K.image_dim_ordering() == 'th':
        concat_axis = 1
    elif K.image_dim_ordering() == 'tf':
        concat_axis = -1
        
    x = merge(block_to_upsample, mode = 'concat', concat_axis = concat_axis)
    print('shape:' + str(x._keras_shape))
    x = Deconvolution2D(n_filters_keep, 3, 3, x._keras_shape, activation = 'linear', border_mode='same', subsample = (2, 2))(x)
    print('shape:' + str(x._keras_shape))
    x = merge([x, skip_connection], mode = 'concat', concat_axis = concat_axis)
    
    return x

#def TransitionUp(skip_connection, block_to_upsample, n_filters_keep):
#    """
#    Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection """
#
#    # Upsample
#    l = ConcatLayer(block_to_upsample)
#    l = Deconv2DLayer(l, n_filters_keep, filter_size=3, stride=2,
#                      crop='valid', W=HeUniform(gain='relu'), nonlinearity=linear)
#    # Concatenate with skip connection
#    l = ConcatLayer([l, skip_connection], cropping=[None, None, 'center', 'center'])
#
#    return l
#    # Note : we also tried Subpixel Deconvolution without seeing any improvements.
#    # We can reduce the number of parameters reducing n_filters_keep in the Deconvolution







