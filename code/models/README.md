# Models

## VGG16 and VGG19 network
[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf).
### Summary
VGG architecture is a widely known convolutional neural network with outstanding results for image classification. It is based on very deep models (up to 19 layers) with small convolutional filters (3x3, 5x5 and even 1x1). On this work, several depths between 16 and 19 layers are evaluated with this small filters. In addition to the new architectures, the work is also focused on studying the effects of normalization (L2, batch normalization and local response normalization), the scaling of the image on training and testing time, the multicropping approach and the fusion of different models. Finally, the VGG model is also proved to worked for localization and to be applied for other datasets.
### Results

[Weights file]() | [Configuration file]() | [Implementation](vgg.py)

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
Rethinking the Inception Architecture for Computer Vision
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

[Weights file (for the first result)](https://drive.google.com/file/d/0ByoayY6Lo-XTT1A2MjV5a0VTWlk/view?usp=sharing) | [Configuration file](../config/dense_tt100k_scratch.py) | [Implementation](densenetFCN.py)
