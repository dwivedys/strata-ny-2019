# Copyright 2019 Cloudera, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# # TensorFlow neural network model for image classification

# This example uses TensorFlow's 
# [Keras API](https://www.tensorflow.org/guide/keras)
# to classify chess pieces based on images.


# ## 0. Preliminaries

# Import the required modules
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import array_to_img
from IPython.display import display

# Import the functions defined in `load_images.py` 
# in the directory `2_machine_learning`
sys.path.append('2_machine_learning')
from load_images import load_labeled_images, load_unlabeled_images


# ## 1. Load data

# Specify the root directory where the images are
img_root = 'data/chess/images'

# There are images of pieces from four different chess
# sets (A, B, C, and D); specify which one(s) to use
chess_sets = ['A','B','C','D']

(features, labels) = \
  load_labeled_images(img_root, chess_sets)

# See one of the images and its label
array_to_img(features[0])
labels[0]


# ## 2. Prepare data

# Encode the character string labels as integer
# numbers using scikit-learn's 
# [`LabelEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
label_encoder = LabelEncoder()
label_encoder.fit(labels)
labels_encoded = label_encoder.transform(labels)

# Split the features and labels each
# into training and test sets
train_x, test_x, train_y, test_y = \
  train_test_split(features,labels_encoded,test_size=0.2)


# ## 3. Specify model

# With the Keras API, you can specify a model as 
# a stack of layers. This example specifies a 
# convolutional neural network. In addition to dense
# (fully connected) layers, this type of neural network
# also uses:
# - Convolutional layers (for filtering and weighting)
# - Pooling layers (for downsampling) 

# These types of layers allow a neural network to 
# differentiate between images based on subregions, 
# and efficiently learn what visual features are most
# important for predicting labels.
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(units=512, activation='relu'),
  tf.keras.layers.Dense(units=128, activation='relu'),
  tf.keras.layers.Dense(units=6, activation='softmax')
])

# Before training a Keras model, you need to
# configure it using the `compile` method,
# specifying an optimizer, a loss function, and
# one or more metrics
model.compile(
  optimizer='rmsprop',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)


# ## 4. Train model

# Call the `fit` method to train the model
model.fit(train_x, train_y, epochs=15)


# ## 5. Evaluate model

# Call the `evaluate` method to evaluate (test)
# the trained model
model.evaluate(test_x, test_y)


# ## 6. Make predictions

# Use the trained model to generate predictions
# on unlabeled images from different chess sets

pred_features = \
  load_unlabeled_images(img_root, ['unknown'])

# Call the `predict` method to return class 
# probabilities
pred_probs = model.predict(pred_features)
pred_probs

# Call the `predict_classes` method to return
# predicted classes, then use the `inverse_transform`
# method of the `LabelEncoder` to decode these
# to the names of chess pieces
pred_classes = model.predict_classes(pred_features)
pred_labels = label_encoder.inverse_transform(pred_classes)

# Display the images and predicted labels
for (x, y) in zip(pred_features, pred_labels):
  display(array_to_img(x))
  print(y)
