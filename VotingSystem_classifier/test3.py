  # -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import scipy
import numpy as np
import argparse

# Same network definition as before
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep)
network = conv_2d(network, 96, 5, activation='relu')
network = max_pool_2d(network, 3)
network = conv_2d(network, 192, 5, activation='relu')
network = max_pool_2d(network, 3)
network = conv_2d(network, 192, 3, activation='relu')
network = conv_2d(network, 192, 1, activation='relu')
network = conv_2d(network, 10, 1, activation='relu')
network = avg_pool_2d(network,6)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='sgd',
                     loss='categorical_crossentropy',
                     learning_rate=0.25)

model = tflearn.DNN(network, tensorboard_verbose=3, checkpoint_path='model3.tfl.ckpt')
model.load("model3.tfl.ckpt-25000")

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype('float32')
file = open("voting3.txt","w") 
for img in x_test:
	prediction = model.predict([img])
	sorted_p = sorted(((value, index) for index, value in enumerate(prediction[0])), reverse=True)
	file.write(str(sorted_p[0][1]))
	file.write('\n')
file.close()

