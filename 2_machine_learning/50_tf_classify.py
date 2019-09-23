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

# # TensorFlow neural network model for multi-class classification

# This example uses TensorFlow's 
# [Keras API](https://www.tensorflow.org/guide/keras)
# to classify chess pieces based on measurements.


# ## 0. Preliminaries

# Import the required modules
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ## 1. Load data

# Load data representing one brand of chess set ("set A")
chess = pd.read_csv('data/chess/one_chess_set.csv')

# View the data
chess


# ## 2. Prepare data

# Encode the character string labels as integer
# numbers using scikit-learn's 
# [`LabelEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
label_encoder = LabelEncoder()
label_encoder.fit(chess.piece)

chess_encoded = chess.assign(
  piece = label_encoder.transform(chess.piece)
)

# Separate the features (x) and labels (y)
features = chess_encoded \
  .filter(['base_diameter','height','weight'])

labels = chess_encoded \
  .filter(['piece'])

# Split the features and labels each
# into training and test sets
train_x, test_x, train_y, test_y = train_test_split(
  features,
  labels,
  test_size=0.2
)


# ## 3. Specify model

# With the Keras API, you can specify a model as 
# a stack of layers. This example specifies a
# feed-forward neural network for 6-class
# classification, with all dense (fully connected)
# layers. The network has two hidden layers, each
# with 10 nodes (cells)
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dense(6, activation='softmax')
])

# Before training a Keras model, you need to
# configure it using the `compile` method,
# specifying an optimizer, a loss function, and
# one or more metrics
model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)


# ## 4. Train model

# Call the `fit` method to train the model
model.fit(
  train_x.values,
  train_y.values,
  epochs=200
)


# ## 5. Evaluate model

# Call the `evaluate` method to evaluate (test)
# the trained model
model.evaluate(test_x.values, test_y.values)


# ## 6. Make predictions

# Use the model to generate predictions based on
# the features of three unlabeled chess pieces
# from "set A":
new_data = pd.DataFrame({
  'base_diameter':[37.4, 35.8, 31.7],
  'height':[97.2, 76.4, 46.6],
  'weight':[52.4, 46.1, 34.6],
})

# Call the `predict` method to return class 
# probabilities
model.predict(new_data.values)

# Call the `predict_classes` method to return
# predicted classes, then use the `inverse_transform`
# method of the `LabelEncoder` to decode these
# to the names of chess pieces
predictions = model.predict_classes(new_data.values)
label_encoder.inverse_transform(predictions)


# ## Exercises

# 1. This code trains the model using measurements of
#    pieces from just one brand of chess set ("set A").
#    In the **1. Load data** section, modify the code
#    to load data with measurements from four different
#    brands of chess sets (A, B, C, and D). This data
#    is in the file `four_chess_sets.csv`. How does 
#    this affect the accuracy of the model on the test 
#    (evaluation) set?

# 2. Why did the accuracy change?

# 3. In the **3. Specify model** section, try
#    increasing the number of nodes in the hidden
#    layers and adding more hidden layers to improve
#    the accuracy of the model.

# 4. What else you could do to try to improve the
#    accuracy of the model?
