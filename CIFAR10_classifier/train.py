import numpy as np
from numpy import array
import keras
import tflearn
from keras.datasets import cifar10
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

num_classes = 10

# Load CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Make dataset input float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Shuffle the data
x_train, y_train = shuffle(x_train, y_train)

# Make sure the data is normalized
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep)

# Step 1: Convolution
network = conv_2d(network, 32, 3, activation='relu')

# Step 2: Max pooling
network = max_pool_2d(network, 2)

# Step 3: Convolution again
network = conv_2d(network, 64, 3, activation='relu')

# Step 4: Convolution yet again
network = conv_2d(network, 64, 3, activation='relu')

# Step 5: Max pooling again
network = max_pool_2d(network, 2)

# Step 6: Fully-connected 512 node neural network
network = fully_connected(network, 512, activation='relu')

# Step 7: Fully-connected neural network with 10 outputs
network = fully_connected(network, 10, activation='softmax')

network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
# model object
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='cifar-classifier.tfl.ckpt')

model.fit(x_train, y_train, n_epoch=50, shuffle=True, validation_set=(x_test, y_test),
          show_metric=True, batch_size=100,
          snapshot_epoch=True,
          run_id='cifar-classifier')

# Save model when training is complete to a file
model.save("cifar-classifier.tfl")
print("Network trained and saved as cifar-classifier.tfl!")
