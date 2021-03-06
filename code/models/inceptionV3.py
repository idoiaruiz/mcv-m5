# Keras imports

from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation

from keras.applications.inception_v3 import InceptionV3

#paper: https://arxiv.org/abs/1512.00567

def build_inceptionV3(img_shape=(3, 299, 299), n_classes=1000,  l2_reg=0.,
                load_imageNet=False, freeze_layers_from='base_model'):


    # Decide if load pretrained weights from imagenet
    if load_imageNet:
        weights = 'imagenet'
    else:
        weights = None

    base_model = InceptionV3(include_top=True, weights=weights, input_tensor=None, input_shape=None)

    x = base_model.layers[-2].output
    x = Dense(n_classes, name='dense_Roque')(x)
    predictions = Activation("softmax", name="softmax")(x)


    # This is the model we will train
    model = Model(input=base_model.input, output=predictions)

    # Freeze some layers
    if freeze_layers_from is not None:
        if freeze_layers_from == 'base_model':
            print ('   Freezing base model layers')
            for layer in base_model.layers:
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
