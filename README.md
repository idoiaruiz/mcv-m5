# Team 5
Master in Computer Vision - M5 Visual recognition.

Professors: [dvazquezcvc](https://github.com/dvazquezcvc) and [lluisgomez](https://github.com/lluisgomez).

## Abstract

## Report

Overleaf document for the  [report](https://www.overleaf.com/read/pkxqmvsfjwqm)

## Project slides
Google slides for the [project](https://drive.google.com/open?id=1xjIemmBNH8XuA9MFeBLiE718U4IJnI8zhXV-gAfD86o) 
## Contributors

 * [Idoia Ruiz](https://github.com/idoiaruiz) (idoiaruizl@gmail.com)
 * [Roque Rodriguez](https://github.com/RoqueRouteiral) (roque.rodriguez.outeiral@gmail.com)
 * [Lidia Talavera](https://github.com/LidiaTalavera) (talaveramartinez.l@gmail.com)
 * [Onofre Martorell](https://github.com/OnofreMartorell) (onofremartorelln@gmail.com)

 
## ResNet
The ResNet is a framework that comes from a modification of the common Convolutional Neural Network. The main idea of this framework is the following: if we want to approximate a mapping H(x) using a stack of layers, we can approximate directly the mapping, or we can approximate F(x) = H(x) - x and add x at the end of the network to obtain the original mapping as output. The new mapping F(x) is the residual function and using it, we can write H(x) = F(x) + x. This idea is motivated by the fact that deeper networks has greater training and test error.

The authors of the use this idea to create a new CNN architecture: they create a bottleneck architecture where each block (corresponding with a residual function) has a 1x1, 3x3 and 1x1 convolutional layers. They tried different depths, going from 50 to 152 layers.

They compared they new architecture with the state-of-the-art architectures, as GoogleNet and VGG and they obtain less test error wih the 152-layer ResNet.