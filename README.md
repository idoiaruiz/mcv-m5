# Scene Understanding for Autonomous Vehicles
Master in Computer Vision - M5 Visual recognition.

Professors: [dvazquezcvc](https://github.com/dvazquezcvc) and [lluisgomez](https://github.com/lluisgomez).

Team: *Living dead students* :mortar_board:

## Abstract
Applying and analysing deep learning state of the art techniques, we perform object detection, recognition, and semantic segmentation, evaluated on images from urban driving [datasets](https://github.com/idoiaruiz/mcv-m5/tree/master/code#available-dataset-wrappers).

## Tasks progress :chart_with_upwards_trend:
### Object recognition
For the object recognition problem, we implement and test several architectures, training them from scratch as well as fine-tuning using pretrained weights. We also boost the performance of the networks using different pre-processing techniques, and performing data augmentation and hyperparameter optimization.
 - [x] Implement ResNet architecture and train it both from scratch and fine-tuning using ImageNet weights
 - [x] Implement InceptionV3 architecture and train it both from scratch and fine-tuning using ImageNet weights
 - [x] Implement DenseNet and train it from scratch. Test the use of Dropout layers.
 - [x] Transfer learning between TT100K and BelgiumTSC datasets using the VGG model.
 - [ ] Try several pre-processing methods for the TT100K dataset with VGG model.
 - [ ] Evaluate crop vs resize for the input images for the VGG model with TT100K dataset.
 - [ ] Boost the performance of the VGG model using data augmentation, bagging as well as optimizing the hyperparameters.

## Documents :clipboard:
- Overleaf document for the  [report](https://www.overleaf.com/read/pkxqmvsfjwqm)
- Google [slides](https://drive.google.com/open?id=1xjIemmBNH8XuA9MFeBLiE718U4IJnI8zhXV-gAfD86o)
- [Summaries and results](https://github.com/idoiaruiz/mcv-m5/tree/master/code/models#models) for the used models

## Contributors :couple::couple:
 * [Idoia Ruiz](https://github.com/idoiaruiz) (idoiaruizl@gmail.com)
 * [Roque Rodriguez](https://github.com/RoqueRouteiral) (roque.rodriguez.outeiral@gmail.com)
 * [Lidia Talavera](https://github.com/LidiaTalavera) (talaveramartinez.l@gmail.com)
 * [Onofre Martorell](https://github.com/OnofreMartorell) (onofremartorelln@gmail.com)
