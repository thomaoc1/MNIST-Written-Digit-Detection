# MNIST Written Digit Recognition
The goal of this mini project is to use different machine learning algorithms to recognize handwritten digits. The dataset used is the [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) which is a collection of 28x28 pixel images of handwritten digits. The dataset contains 60,000 training images and 10,000 testing images. The goal is to correctly identify digits from 0 to 9. 

The following sections will describe the preprocessing performed followed by the different methods used thus far for this task.

## Preprocessing
The dataset is normalised to have values from 0 to 1. The images are also flattened to be 1D arrays of 784 elements.

## Methods
### Classic Models
These models share a lot of the same logic thus most of their logic is generalised in the [ClassicModel](https://github.com/thomaoc1/MNIST-digit-detection/blob/main/src/classicmodels/classicmodel.py) class.
#### 1. [(Non-)Linear Support Vector Classifiers (SVC)](https://github.com/thomaoc1/MNIST-digit-detection/tree/main/src/classicmodels/svc)
#### 2. [Random Forest ](https://github.com/thomaoc1/MNIST-digit-detection/tree/main/src/classicmodels/randforest)
#### 3. [K-Nearest Neighbours](https://github.com/thomaoc1/MNIST-digit-detection/tree/main/src/classicmodels/knn)

### Neural Network Models
#### 4. [Convolutional Neural Network (CNN)](https://github.com/thomaoc1/MNIST-digit-detection/tree/main/src/cnn)

## Evaluations 
The evaluation of each model is done in a jupyter notebook found in each model's respective directory. They can be found here:

1. [Linear SVC](https://github.com/thomaoc1/MNIST-digit-detection/blob/main/src/classicmodels/svc/linear/evaluation.ipynb)
2. [Non-Linear SVC](https://github.com/thomaoc1/MNIST-digit-detection/blob/main/src/classicmodels/svc/nonlinear/evaluation.ipynb)
3. [Random Forest](https://github.com/thomaoc1/MNIST-digit-detection/blob/main/src/classicmodels/randforest/evaluation.ipynb)
4. [KNN](https://github.com/thomaoc1/MNIST-digit-detection/blob/main/src/classicmodels/knn/evaluation.ipynb)
5. [CNN](https://github.com/thomaoc1/MNIST-digit-detection/blob/main/src/cnn/evaluation.ipynb)

### Features
For the CNN, you can input a self drawn digit as shown below and see a prediction in real time.

![example.gif](res%2Fexample.gif)