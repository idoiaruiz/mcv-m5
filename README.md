# Deep Learning for scene understanding
Master in Computer Vision - M5 Visual recognition.

Professors: [dvazquezcvc](https://github.com/dvazquezcvc) and [lluisgomez](https://github.com/lluisgomez).

Team: *Living dead students* :mortar_board:

## Abstract
Applying and analysing deep learning state of the art techniques, we perform object detection, recognition, and semantic segmentation, evaluated on images from urban driving [datasets](https://github.com/idoiaruiz/mcv-m5/tree/master/code#available-dataset-wrappers).

## Tasks progress :chart_with_upwards_trend:
### Object recognition
For the object recognition problem, we implement and test several architectures, training them from scratch as well as fine-tuning using pretrained weights. We also boost the performance of the networks using different pre-processing techniques, and performing data augmentation and hyperparameter optimization.
 - [x] Implement ResNet architecture and train it both from scratch and fine-tuning using ImageNet weights.
 - [x] Implement InceptionV3 architecture and train it both from scratch and fine-tuning using ImageNet weights.
 - [x] Implement DenseNet and train it from scratch. Test the use of Dropout layers.
 - [x] Train VGG for TT100K, BelgiumTSC and KITTI datasets from scratch.
 - [x] Transfer learning between TT100K and BelgiumTSC datasets using the VGG model.
 - [x] Try several pre-processing methods for the TT100K dataset with VGG model.
 - [x] Evaluate crop vs resize for the input images for the VGG model with TT100K dataset.
 - [x] Boost the performance of the VGG model using data augmentation, bagging as well as optimizing the hyperparameters.

### Object detection
For object detection, we train and test the YOLOv2 model using the ImageNet pretrained weights. We also implement the SSD model and boost the performance of the networks with pre-processing, hyperparameter optimization and data augmentation.
 - [x] Implement the SSD architecture and train it from scratch for the Udacity and TT100k datasets.
 - [x] Train YOLOv2 and Tiny-YOLO models for the TT100K and Udacity datasets using the ImageNet pretrained weights.
 - [x] Boost the performance for the YOLOv2 model with preprocessing techniques and data augmentation.
 - [x] Analyze the Udacity dataset and propose two approaches for dealing with the differences between the validation and test datasets. Test it with the YOLOv2 model.
 - [x] Integrate F-score and FPS evaluation in our framework and evaluate YOLOv2 and Tiny-YOLO models.

### Semantic segmentation
We implement several state-of-the-art semantic segmentation architectures, training them for the Camvid dataset. We train as well the FCN model for the Synthia dataset. This model is also boosted with hyperparameter optimization and data augmentation.
- [x] Implement the Segnet model (Segnet with VGG and the 'Segnet Basic' version) and train them from scratch for the Camvid dataset.
- [x] Train the FCN model for the Camvid and Synthia datasets from scratch.
- [x] Boost the performance for the FCN and SegNet models with preprocessing techniques and data augmentation.
- [x] Implement a semantic segmentation architecture using DenseNet as the classification architecture.
- [ ] Implement a semantic segmentation architecture using InceptionV3 as the classification architecture.

## Usage :computer:
1. Fix the paths for the datasets in [train.py](code/train.py) for working on your machine.

2. Run the code
```
python train.py -c config/dataset.py -e expName
```
   where ```dataset.py``` is the [configuration file](https://github.com/idoiaruiz/mcv-m5/tree/master/code/models#results) for this test, and ```expName``` is the name of the directory where the results are saved.

## Documents :clipboard:
- Overleaf document for the  [report](https://www.overleaf.com/read/pkxqmvsfjwqm)
- Google [slides](https://drive.google.com/open?id=1xjIemmBNH8XuA9MFeBLiE718U4IJnI8zhXV-gAfD86o)
- [Summaries, results and trained weights](https://github.com/idoiaruiz/mcv-m5/tree/master/code/models#models) for the used models

## Contributors :couple::couple:
 * [Idoia Ruiz](https://github.com/idoiaruiz) (idoia.ruizl@e-campus.uab.cat)
 * [Roque Rodriguez](https://github.com/RoqueRouteiral) (roque.rodriguez.outeiral@gmail.com)
 * [Lidia Talavera](https://github.com/LidiaTalavera) (talaveramartinez.l@gmail.com)
 * [Onofre Martorell](https://github.com/OnofreMartorell) (onofremartorelln@gmail.com)
