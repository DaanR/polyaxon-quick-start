#!/usr/bin/python
#
# Copyright 2018-2020 Polyaxon, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import keras
import tensorflow as tf

# load dogs vs cats dataset, reshape and save to a new file
from os import listdir
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dense, Dropout, Activation, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers

from polyaxon import tracking
from polyaxon.tracking.contrib.keras import PolyaxonKerasCallback, PolyaxonKerasModelCheckpoint
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import numpy as np
from matplotlib import image
from matplotlib import pyplot

OPTIMIZERS = {
    'adam': optimizers.Adam,
    'rmsprop': optimizers.RMSprop,
    'sgd': optimizers.SGD,
}


# load image as pixel array
image = image.imread('kaggle_bee_vs_wasp/bee1/1240800_e5f2b40032_n.jpg')
# summarize shape of the pixel array
print(image.dtype)
print(image.shape)

photos, labels = list(), list()
folder = 'kaggle_bee_vs_wasp/bee1/'
# enumerate files in the directory
for file in listdir(folder):
	# determine class
	output = 1.0
	# load image
	photo = load_img(folder + file, target_size=(200, 200))
	# convert to numpy array
	photo = img_to_array(photo)
	# store
	photos.append(photo)
	labels.append(output)
# convert to a numpy arrays
photos = np.asarray(photos)
labels = np.asarray(labels)
print(photos.shape, labels.shape)


def create_model(
    conv1_size,
    conv2_size,
    dropout,
    hidden1_size,
    conv_activation,
    dense_activation,
    optimizer,
    learning_rate,
    loss,
    num_classes,
):
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = optimizers.SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--conv1_size',
        type=int,
        default=32)
    parser.add_argument(
        '--conv2_size',
        type=int,
        default=64
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2
    )
    parser.add_argument(
        '--hidden1_size',
        type=int,
        default=500
    )
    parser.add_argument(
        '--conv_activation',
        type=str,
        default="relu"
    )
    parser.add_argument(
        '--dense_activation',
        type=str,
        default="relu"
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10
    )
    parser.add_argument(
        '--loss',
        type=str,
        default="categorical_crossentropy"
    )

    args = parser.parse_args()


    # Data
    photos = photos.astype('float32')
    photos /= 255.


    # Polyaxon
    tracking.init()
    plx_callback = PolyaxonKerasCallback()
    plx_model_callback = PolyaxonKerasModelCheckpoint()
    log_dir = tracking.get_tensorboard_path()

    print("log_dir", log_dir)
    print("model_dir", plx_model_callback.filepath)
    # TF Model
    model = create_model(
        conv1_size=args.conv1_size,
        conv2_size=args.conv2_size,
        dropout=args.dropout,
        hidden1_size=args.hidden1_size,
        conv_activation=args.conv_activation,
        dense_activation=args.dense_activation,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        loss=args.loss,
        num_classes=2
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq=100
    )

    model.fit(x=photos,
              y=labels,
              epochs=args.epochs,
              callbacks=[tensorboard_callback, plx_callback, plx_model_callback])
