# Image Recognition Research

## 1. Introduction

The objective of this project is to categorize images into 10 product types and 7 brands.

## 2. Data Preprocessing

### a) Reading Images

In the file `data_preprocessing.py`, a Python class is created to perform the following data preprocessing tasks:

- Read images:
  - Utilize the `imread` method from the `cv2` module to read images into a 3-dimensional array.
  - Use the `load_img` method from the `keras` module to read images as PIL images.

### b) Augmentation

- Randomly adjust the light and contrast of images by processing their 3-dimensional arrays.
- Generate rotated, cropped, and zoomed versions of the original images.

### c) Normalization

- Resize the images.
- Convert the images to grayscale using the `convert` method from the PIL package.

### d) Vectorization

- Vectorize the images and perform one-hot encoding on the labels.

### e) Train-Test Separation

## 3. Machine Learning Models

- Self-trained 5-layer CNN model (baseline).
- Combination of HoG (Histogram of Oriented Gradient) and SVM.
- Fine-tune a pre-trained network, "inception_v1" or GoogleNet, a 22-layer deep learning neural network.
  - Implemented with the final branch (AdamOptimizer with learning rate = 0.0001 to minimize softmax cross-entropy).
  - [Google's Inception V1 model](http://alpha.tfhub.dev/google/imagenet/inception_v1/classification/1) is utilized.
