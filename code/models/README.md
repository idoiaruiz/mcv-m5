# Models

## VGG16 and VGG19 network
[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf).
### Summary
VGG architecture is a widely known convolutional neural network with outstanding results for image classification. It is based on very deep models (up to 19 layers) with small convolutional filters (3x3, 5x5 and even 1x1). On this work, several depths between 16 and 19 layers are evaluated with this small filters. In addition to the new architectures, the work is also focused on studying the effects of normalization (L2, batch normalization and local response normalization), the scaling of the image on training and testing time, the multicropping approach and the fusion of different models. Finally, the VGG model is also proved to worked for localization and to be applied for other datasets.
### Results

We first train VGG16 with the preconfigured experiment file for the TT100K dataset obtaining a validation accuracy of 88.63 % and a test accuracy of 95.8%. After that, we evaluate different techniques in the configuration file to see how it affects to the behavior of the network.

[Weights file](https://drive.google.com/open?id=0B06nnAKc0eZvWXhNVXZpdER6ZzQ) | [Configuration file](../config/Vgg16.py) | [Implementation](vgg.py)

In order to perform a good comparison of the crop technique with the resize one, first we resize the image to 256,256 and then crop it to 224,224, as the images from the dataset are smalle

| Train with TT100K dataset | Validation accuracy (%) |Test accuracy (%)|
| ------------- | ------------- |----------------------|
| Vgg16 resize     |     88.63    |         95.80        |
| Vgg16 crop | 88.33 |   96.73  |

[Configuration file - train Vgg16 crop](../config/tt100ktrainVggCrop.py) | [Configuration file - test Vgg16 crop](../config/tt100ktestVggCrop.py)
| [Configuration file - Vgg16 resize](../config/Vgg16.py)

Then some pre-processing techniques are applied in order to normalize the data. In this case we obtain good results by dividing the std from the dataset and also we test the preprocess of imageNet.

| Train with TT100K dataset | Validation accuracy (%) |Test accuracy (%)|
| ------------- | ------------- |----------------------|
| Division of the std from the dataset     |     92.01    |         96.11        |
| ImageNet norm. | 70.52 |  91.67  |

[Configuration file Std division from dataset](../config/tt100kNormalization.py) | [Configuration file ImageNet norm.](../config/tt100kImageNetNormalization.py)

Once we have trained the VGG16 model with the TT100K dataset, we use those trained weights with the BelgiumTSC dataset. We also train VGG16 from scratch with the BTSC dataset, as well as for the KITTI database, although we do not have evaluation results for KITTI since there was no test set available.


| BelgiumTS dataset and VGG16 | Validation accuracy (%) |Test accuracy (%)|
| ------------- | ------------- |----------------------|
| Transfer learning from Tt100K dataset  |     97.10    |         97.10       |
| Training from scratch | 96.03  |  95,91  |

[Configuration file transfer learning from Tt100K dataset-train ](../config/transferLearningtrain.py) | [Configuration file transfer learning from Tt100K dataset-test ](../config/transferLearningtest.py) | [Configuration train file training from scratch](../config/BelgiumTSscratch.py)

Finally, in order to boost the performance of our network we have done data augmentation with horizontal flipping and bagging.

| Train with TT100K dataset | Validation accuracy (%) |Test accuracy (%)|
| ------------- | ------------- |----------------------|
| Data augmentation  |     91.82    |         93.96       |
| Bagging | 43.19  |  61.85  |

[Configuration file data augmentation](../config/DataAugmentation.py) | [Configuration file bagging](../config/Bagging.py)


## ResNet
[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
### Summary
ResNet is a framework that comes from a modification of the common Convolutional Neural Network. The main idea of this framework is that for approximating a mapping H(x) using a stack of layers, we can directly approximate the mapping, or we can also approximate F(x) = H(x) - x, adding x at the end of the network, to obtain the original mapping as output. The new mapping F(x) is the residual function. With it, the mapping function can be written as H(x) = F(x) + x. This is motivated by the fact that deeper networks have greater training and test error.

The authors use this idea to create a new CNN architecture. They create a bottleneck architecture where each block (that corresponds to a residual function) has 1x1, 3x3 and 1x1 convolutional layers. Trying different depths, going from 50 up to 152 layers, and comparing this new architecture with the state-of-the-art architectures, as GoogleNet and VGG, the authors obtain a smaller test error using the 152-layer ResNet.
### Results
ResNet model is trained from Scratch and with finetunning from the weights of Imagenet. It is seen how for the second case, the loss is not behaving properly. Thus, the learning rate is raise and the results improve. The model is build with the convolutional block from keras and with the fully connected block of the paper.

| Train with TT100K dataset | Validation accuracy (%) |Test accuracy (%)|
| ------------- | ------------- |----------------------|
| ResNet from scratch      |   90.34        |      92.53           |
| ResNet with finetunning with Imagenet (LR = 0.0001) | 78.06 | 79.40     |
| ResNet with finetunning with Imagenet (LR = 0.001) |  78.89   |   83.41   |

[Weights file](https://drive.google.com/open?id=0B1fN3dKxIN8CUkVGQnh5SVFCVVE) | [Configuration file](../config/tt100kResnetFromScratch.py) | [Implementation](resnet.py)

## InceptionV3
[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)
### Results
Inception V3 is trained from scratch and with finetunning from the weights of Imagenet. The model is build with the convolutional block from keras and with the fully connected block of the paper.

| Train with TT100K dataset | Validation accuracy (%) |Test accuracy (%)|
| ------------- | ------------- |----------------------|
| InceptionV3 from scratch      |     94.82     |         97.39        |
| InceptionV3 with finetunning with Imagenet | 92.13 |   98.14  |

[Weights file](https://drive.google.com/open?id=0B1fN3dKxIN8CRjUzMHVkN01QUHM) | [Configuration file](../config/tt100kInceptionFromScratch.py) | [Implementation](inceptionV3.py)

## DenseNet
[Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
### Results
We train our DenseNet model from scratch with the TT100K dataset.
We also test to apply Dropout layers after every convolutional layer setting the dropout rate to 0.2.


| Train from scratch TT100K dataset | Validation accuracy (%) |Test accuracy (%)|
| ------------- | ------------- |----------------------|
| DenseNet      | 84.27         |      91.10           |
| DenseNet with Dropout (rate=0.2) | 68.31 | 64.84     |

[Weights file (for the first result)](https://drive.google.com/file/d/0ByoayY6Lo-XTT1A2MjV5a0VTWlk/view?usp=sharing) | [Configuration file](../config/tt100k_classif_densenet_scratch.py) | [Implementation](densenetFCN.py)

## Yolo
[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
### Summary
As other deep architectures for object detection need to look several times on the image, this architecture only needs to look one time at an image in order to find the objects in it. The architecture divides the image in a 7x7 grid and it passes each cell through a convolutional neural network in order to obtain a bounding box and a list of probabilities for all the classes.

### Results
We train Yolo with the preconfigured experiment file for the TT100K and Udacity datasets. After that, we evaluate, for both datasets, different techniques in the configuration file to see how it affects to the behavior of the network. All the results can be seen the following tables.

#### TT100K Dataset
[Weights file](https://drive.google.com/open?id=0B06nnAKc0eZvZmpiaFRHbXI4Vzg) | [Configuration file](../config/yoloBaselineTt100k.py) | [Implementation](yolo.py)


|  | Precission (%) | Recall (%) | F-score(%) | FPS |
|-------------| ------------- | ------------- |----------------------|----------------------|
|Yolo|    80.78   | 60.57         |      69.23           | 19.90 |
|Tiny Yolo|  53.29 | 27.04 | 35.88     | 31.76 |
|Horizontal and Vertical shift (20%)|  74.36 | 61.04 | 67.04     | 19.65 |
|Horizontal flip|  81.60 | 67.30 | 73.76     | 19.76 |
|Feature Normalization|  81.02 | 69.20 | 74.65     | 19.90 |
|Horizontal flip and feature norm.|  84.89 | 71.83 | 77.81     | 19.78 |

[Configuration file Yolo](../config/yoloBaselineTt100k.py) |
[Configuration file Tiny Yolo](../config/TinyyoloTt100k.py) |
[Configuration file Horizontal and Vertical shift (20%)](../config/yoloHandVshiftTt100k.py) |
[Configuration file Horizontal flip](../config/yoloDataAugTt100k.py) |
[Configuration file Feature normalization](../config/yoloFeatNormTt100k.py) |
[Configuration file Horizontal flip and feature norm.](../config/yoloDataAugFeatNormTt100k.py)

#### Udacity Dataset

[Weights file](https://drive.google.com/open?id=0B06nnAKc0eZvSTlQQVdqdjFqLTQ) |[Configuration file](../config/yoloBaselineUdacity.py) | [Implementation](yolo.py)



|  | Precission (%) | Recall (%) | F-score(%) | FPS |
|-------------| ------------- | ------------- |----------------------|----------------------|
|Yolo|    51.05   | 28.69         |      36.74           | 20.00 |
|Tiny Yolo|  11.02 | 4.11 | 5.98     | 32.00 |
|Horizontal and Vertical shift (20%)|  53.35 | 32.98 | 40.76     | 19.84 |
|Horizontal flip|  35.76 | 22.51 | 27.63     | 19.94 |
|Feature Normalization|  40.17 | 25.06 | 30.87     | 19.84 |
|Horizontal flip and feature norm.|  40.59 | 22.23 | 28.72     | 19.89 |

[Configuration file Yolo](../config/yoloBaselineUdacity.py) |
[Configuration file Tiny Yolo](../config/TinyyoloUdacity.py) |
[Configuration file Horizontal and Vertical shift (20%)](../config/yoloHandVshiftUdacity.py) |
[Configuration file Horizontal flip](../config/yoloDataAugUdacity.py) |
[Configuration file Feature normalization](../config/yoloFeatNormUdacity.py) |
[Configuration file Horizontal flip and feature norm.](../config/yoloDataAugFeatNormUdacity.py)



## SSD
### Results

We trained the SSD model from scrath for the TT100K and Udacity datasets. However the metrics are still not implemented to evaluate the model.

TT100k dataset

[Configuration file](../config/tt100k_detection_ssd.py) | [Weights file](https://drive.google.com/file/d/0ByoayY6Lo-XTQ2RaZWFQR1VSU0k/view?usp=sharing)

Udacity dataset

[Configuration file](../config/udacity_detection_ssd.py) | [Weights file](https://drive.google.com/file/d/0ByoayY6Lo-XTWlNwQXVCNEg2Q0k/view?usp=sharing)
