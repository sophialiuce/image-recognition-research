# image-recognition-research
**1. Introduction**

The task of this project is to classify images into 10 product types and 7 brands.

**2. Data Preprocessing**

In the file "data_preprocessing.py", I created a Python class to do the following data preprocessing tasks:

a) Read images
* Read images as a 3-dimensional array using “imread” method in module “cv2”.
* Read images as PIL images using “load_img” method from module “keras”.

b) Augmentation
* Randomly change the light and contrast of images by processing the 3-dimensional array of an image.
* Generate rotated, cropped, and zoomed images of the original images.

c) Normalization
* Resize the images
* Grayscale the images using the "convert" method from the PIL package.

d) Vectorization
* vectorized the images and one-hot the labels

e) Train, test separation

**3. Machine learning models**
* Self-trained 5-layer CNN model - baseline.
* Combination of HoG (Histogram of Oriented Gradient) and SVM.
* Fine-tune pre-trained network, "inception_v1" or GoogleNet, a 22-layer deep learning neural network.
  - Implemented the final branch (AdamOptimizer with learningRate = 0.0001 to minimize softmax_cross_entropy).
  - http://alpha.tfhub.dev/google/imagenet/inception_v1/classification/1

