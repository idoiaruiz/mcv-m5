import os

# Keras imports
from metrics.metrics import cce_flatt, IoU, YOLOLoss, YOLOMetrics
from metrics.ssd_training import MultiboxLoss
from keras import backend as K
from keras.utils.visualize_util import plot

# Classification models
#from models.lenet import build_lenet
#from models.alexNet import build_alexNet
from models.vgg import build_vgg
from models.resnet import build_resnet50
from models.inceptionV3 import build_inceptionV3
from models.densenet import build_densenet

# Detection models
from models.yolo import build_yolo
from models.ssd import build_ssd

# Segmentation models
from models.fcn8 import build_fcn8
#from models.unet import build_unet
from models.segnet import build_segnet
#from models.resnetFCN import build_resnetFCN
#from models.densenetFCN import build_densenetFCN

from models.densenet_segmentation import build_densenet_segmentation
#from models.inceptionFCN import build_inceptionFCN

# Adversarial models
#from models.adversarial_semseg import Adversarial_Semseg

from models.model import One_Net_Model

from keras.models import model_from_json

# Build the model
class Model_Factory():
    def __init__(self):
        pass

    # Define the input size, loss and metrics
    def basic_model_properties(self, cf, variable_input_size):
        # Define the input size, loss and metrics
        if cf.dataset.class_mode == 'categorical':
            if K.image_dim_ordering() == 'th':
                in_shape = (cf.dataset.n_channels,
                            cf.target_size_train[0],
                            cf.target_size_train[1])
            else:
                in_shape = (cf.target_size_train[0],
                            cf.target_size_train[1],
                            cf.dataset.n_channels)
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        elif cf.dataset.class_mode == 'detection':
            if cf.model_name == 'ssd':
                in_shape = (cf.target_size_train[0],
                            cf.target_size_train[1],
                            cf.dataset.n_channels,)
                loss = MultiboxLoss(cf.dataset.n_classes, neg_pos_ratio=2.0).compute_loss
                metrics = None
            else: # YOLO
                in_shape = (cf.dataset.n_channels,
                            cf.target_size_train[0],
                            cf.target_size_train[1])
                loss = YOLOLoss(in_shape, cf.dataset.n_classes, cf.dataset.priors)
                metrics = [YOLOMetrics(in_shape, cf.dataset.n_classes, cf.dataset.priors)]
        elif cf.dataset.class_mode == 'segmentation':
            if K.image_dim_ordering() == 'th':
                if variable_input_size:
                    in_shape = (cf.dataset.n_channels, None, None)
                else:
                    in_shape = (cf.dataset.n_channels,
                                cf.target_size_train[0],
                                cf.target_size_train[1])
            else:
                if variable_input_size:
                    in_shape = (None, None, cf.dataset.n_channels)
                else:
                    in_shape = (cf.target_size_train[0],
                                cf.target_size_train[1],
                                cf.dataset.n_channels)
            loss = cce_flatt(cf.dataset.void_class, cf.dataset.cb_weights)
            metrics = [IoU(cf.dataset.n_classes, cf.dataset.void_class)]
        else:
            raise ValueError('Unknown problem type')
        return in_shape, loss, metrics

    # Creates a Model object (not a Keras model)
    def make(self, cf, optimizer=None):
        if cf.model_name in ['lenet', 'alexNet', 'vgg16', 'vgg19', 'resnet50',
                             'InceptionV3', 'densenet', 'fcn8', 'unet', 'segnet_vgg',
                             'segnet_basic', 'resnetFCN', 'yolo', 'tiny-yolo', 'ssd', 
                             'densenet_segmentation']:
            if optimizer is None:
                raise ValueError('optimizer can not be None')

            in_shape, loss, metrics = self.basic_model_properties(cf, False)
            
            model = self.make_one_net_model(cf, in_shape, loss, metrics,
                                            optimizer)

        elif cf.model_name == 'adversarial_semseg':
            if optimizer is None:
                raise ValueError('optimizer is not None')

            # loss, metrics and optimizer are made in class Adversarial_Semseg
            in_shape, _, _ = self.basic_model_properties(cf, False)
            model = Adversarial_Semseg(cf, in_shape)

        else:
            raise ValueError('Unknown model name')

        # Output the model
        print ('   Model: ' + cf.model_name)
        return model

    # Creates, compiles, plots and prints a Keras model. Optionally also loads its
    # weights.
    def make_one_net_model(self, cf, in_shape, loss, metrics, optimizer):
        # Create the *Keras* model
        if cf.model_name == 'fcn8':
            model = build_fcn8(in_shape, cf.dataset.n_classes, cf.weight_decay,
                               freeze_layers_from=cf.freeze_layers_from,
                               path_weights=cf.load_imageNet)
        elif cf.model_name == 'unet':
            model = build_unet(in_shape, cf.dataset.n_classes, cf.weight_decay,
                               freeze_layers_from=cf.freeze_layers_from,
                               path_weights=None)
        elif cf.model_name == 'segnet_basic':
            model = build_segnet(in_shape, cf.dataset.n_classes, cf.weight_decay,
                                 freeze_layers_from=cf.freeze_layers_from,
                                 basic=True)
        elif cf.model_name == 'segnet_vgg':
            model = build_segnet(in_shape, cf.dataset.n_classes, cf.weight_decay,
                                 freeze_layers_from=cf.freeze_layers_from,
                                 basic=False)
        elif cf.model_name == 'resnetFCN':
            model = build_resnetFCN(in_shape, cf.dataset.n_classes, cf.weight_decay,
                                    freeze_layers_from=cf.freeze_layers_from,
                                    path_weights=None)
        elif cf.model_name == 'inceptionFCN':
            model = build_inceptionFCN(in_shape, cf.dataset.n_classes, cf.weight_decay,
                                    freeze_layers_from=cf.freeze_layers_from,
                                    path_weights=None)									
        elif cf.model_name == 'densenet':
            model = build_densenet(in_shape, cf.dataset.n_classes, cf.weight_decay,
                                      freeze_layers_from=cf.freeze_layers_from,
                                      path_weights=None)
        elif cf.model_name == 'lenet':
            model = build_lenet(in_shape, cf.dataset.n_classes, cf.weight_decay)
        elif cf.model_name == 'alexNet':
            model = build_alexNet(in_shape, cf.dataset.n_classes, cf.weight_decay)
        elif cf.model_name == 'vgg16':
            model = build_vgg(in_shape, cf.dataset.n_classes, 16, cf.weight_decay,
                              load_imageNet=cf.load_imageNet,
                              freeze_layers_from=cf.freeze_layers_from)
        elif cf.model_name == 'vgg19':
            model = build_vgg(in_shape, cf.dataset.n_classes, 19, cf.weight_decay,
                              load_imageNet=cf.load_imageNet,
                              freeze_layers_from=cf.freeze_layers_from)
        elif cf.model_name == 'resnet50':
            model = build_resnet50(in_shape, cf.dataset.n_classes, cf.weight_decay,
                                   load_imageNet=cf.load_imageNet,
                                   freeze_layers_from=cf.freeze_layers_from)
        elif cf.model_name == 'InceptionV3':
            model = build_inceptionV3(in_shape, cf.dataset.n_classes,
                                      cf.weight_decay,
                                      load_imageNet=cf.load_imageNet,
                                      freeze_layers_from=cf.freeze_layers_from)
        elif cf.model_name == 'yolo':
            model = build_yolo(in_shape, cf.dataset.n_classes,
                               cf.dataset.n_priors,
                               load_imageNet=cf.load_imageNet,
                               freeze_layers_from=cf.freeze_layers_from, tiny=False)
        elif cf.model_name == 'tiny-yolo':
            model = build_yolo(in_shape, cf.dataset.n_classes,
                               cf.dataset.n_priors,
                               load_imageNet=cf.load_imageNet,
                               freeze_layers_from=cf.freeze_layers_from, tiny=True)
        elif cf.model_name == 'ssd':
            model = build_ssd(in_shape, cf.dataset.n_classes+1,
                              cf.dataset.n_priors,
                              freeze_layers_from=cf.freeze_layers_from)
        elif cf.model_name == 'densenet_segmentation':
            model = build_densenet_segmentation(in_shape, cf.dataset.n_classes, weight_decay = cf.weight_decay,
                   freeze_layers_from = cf.freeze_layers_from, path_weights = cf.load_imageNet)
        else:
            raise ValueError('Unknown model')

        # Load pretrained weights
        if cf.load_pretrained:
            print('   loading model weights from: ' + cf.weights_file + '...')
            # If the weights are from different datasets
            if cf.different_datasets:
                if cf.freeze_layers_from == 'base_model':
                    raise TypeError('Please, enter the layer id instead of "base_model"'
                          ' for the freeze_layers_from config parameter')
                croppedmodel = model_from_json(model.to_json())
                # Remove not frozen layers
                for i in range(len(model.layers[cf.freeze_layers_from:])):
                    croppedmodel.layers.pop()
                # Load weights only for the frozen layers
                croppedmodel.load_weights(cf.weights_file, by_name=True)
                model.set_weights(croppedmodel.get_weights())
            else:
                model.load_weights(cf.weights_file, by_name=True)

        # Compile model
        model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

        # Show model structure
        if cf.show_model:
            model.summary()
            plot(model, to_file=os.path.join(cf.savepath, 'model.png'))

        # Output the model
        print ('   Model: ' + cf.model_name)
        # model is a keras model, Model is a class wrapper so that we can have
        # other models (like GANs) made of a pair of keras models, with their
        # own ways to train, test and predict
        return One_Net_Model(model, cf, optimizer)
