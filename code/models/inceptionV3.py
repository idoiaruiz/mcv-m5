# Keras imports

from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation

from keras.applications.inception_v3 import InceptionV3

#paper: https://arxiv.org/abs/1512.00567

def build_inceptionV3(img_shape=(3, 299, 299), n_classes=1000,  l2_reg=0.,
                load_pretrained=False, freeze_layers_from='base_model'):
                

    # Decide if load pretrained weights from imagenet
    if load_pretrained:
        weights = 'imagenet'
    else:
        weights = None
        
    #base_model = InceptionV3(include_top=False, weights=weights, input_tensor=None, input_shape=None)
    base_model = InceptionV3(include_top=True, weights=weights, input_tensor=None, input_shape=None)

    # add a global spatial average pooling layer
    x = base_model.layers[-2].output
#    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
#    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 45 classes
    x = Dense(n_classes, name='dense_Roque')(x)
    predictions = Activation("softmax", name="softmax")(x)
            
    
    # This is the model we will train
    model = Model(input=base_model.input, output=predictions)
    
    return model