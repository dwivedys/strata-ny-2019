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

# # TensorFlow convolutional neural network for image classification

# This example applies the estimator defined in the
# file `cnnmodel.py` to a simple image classification 
# task.

# ## 0. Preliminaries

# Import the required modules
import os, random, math, subprocess
import tensorflow as tf
from IPython.display import Image, display

# Import the function `cnn_model_fn` from the file 
# `cnnmodel.py` in the directory `2_machine_learning`
import sys
sys.path.append('2_machine_learning')
from cnnmodel import cnn_model_fn


# ## 1. Load data

# Specify the unique labels (names of the chess pieces)
chess_pieces = ['King', 'Queen', 'Rook', 'Bishop', 'Knight', 'Pawn']

# Specify the root directory where the images are
img_root = 'data/chess/images'

# Make empty lists to hold image file paths (x) and 
# labels (y)
(x, y) = ([], [])

# There are images of pieces from four different chess
# sets (A, B, C, and D); specify which one use
chess_sets = ['A','B','C','D']

# Fill the empty lists with the file paths and labels
for chess_set in chess_sets:
  for chess_piece in chess_pieces:
    img_dir = img_root + '/' + chess_set + '/' + chess_piece + '/'
    img_paths = [img_dir + d for d in os.listdir(img_dir)]
    img_labels = [chess_piece] * len(img_paths)
    x.extend(img_paths)
    y.extend(img_labels)

# View the image file paths and labels
for path, label in zip(x, y):
  print((path, label))


# ## 2. Prepare data

# Split the paths and labels into 80% training, 20% test
train_frac = 0.8
train_n = int(math.floor(train_frac * len(x)))
indices = list(range(0, len(x)))
random.shuffle(indices)
train_indices = indices[0:train_n]
test_indices = indices[train_n:]
train_x = [x[i] for i in train_indices]
train_y = [y[i] for i in train_indices]
test_x = [x[i] for i in test_indices]
test_y = [y[i] for i in test_indices]

# Encode the labels by transforming the names of the
# chess pieces to numeric codes. This is required
# because the estimator implemented in `cnn_model_fn`
# cannot accept a label vocabulary.
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(chess_pieces) # this modifies encoder in place
train_y_encoded = encoder.transform(train_y)
test_y_encoded = encoder.transform(test_y)

# TensorFlow processes records in batches. Set the
# batch size:
BATCH_SIZE = 100

# Define a function that reads an image from a file,
# decodes it to numbers, and returns a two-element tuple 
# `(features, labels)` where `features` is a dictionary
# containing the image pixel data
def _parse_function(path, label):
    image = tf.image.decode_png(tf.read_file(path))
    return ({'image':image}, label)

# Define input functions to supply data for training
# and evaulating the model

# These functions apply `_parse_function`
def train_input_fn():
  dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y_encoded))
  dataset = dataset.map(_parse_function)
  dataset = dataset.shuffle(len(train_x)).repeat().batch(BATCH_SIZE)
  return dataset

def test_input_fn():
  dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y_encoded))
  dataset = dataset.map(_parse_function)
  dataset = dataset.batch(BATCH_SIZE)
  return dataset


# ## 3. Specify model

# Create a list with the feature column
my_feature_columns = [
  tf.feature_column.numeric_column('image', shape=[128, 128])
]

# Instantiate the estimator, specifying the 
# model to use and the feature columns
model = tf.estimator.Estimator(
  model_fn=cnn_model_fn,
  params={
      'feature_columns': my_feature_columns,
  }
)


# ## 4. Train model

# TensorFlow trains the model in multiple steps
# (iterations). In each step, the model learns
# from one batch of training data. You can specify
# the number of training steps that TensorFlow should
# perform before it stops the iterative training
# process.
TRAIN_STEPS = 500

# Call the `train` method to train the model
model.train(
  input_fn=train_input_fn,
  steps=TRAIN_STEPS
)


# ## 5. Evaluate model

# Call the `evaluate` method to evaluate (test) the
# trained model
eval_result = model.evaluate(
  input_fn=test_input_fn
)

# Print the result to examine the accuracy
print(eval_result)


# ## 6. Make predictions

# Use the trained model to generate predictions
# on unlabeled images

# Some of these images are of pieces from other
# chess sets (not from set A)
img_dir = img_root + '/unknown/'
img_paths = [img_dir + d for d in os.listdir(img_dir)]
pred_x = img_paths

# Define a function that reads an image from a file

# This is similar to the `_parse_function` function
# defined above, but without labels
def _predict_parse_function(path):
    image = tf.image.decode_png(tf.read_file(path))
    return ({'image':image})

# Define an input function to supply data for generating
# predictions
def predict_input_fn():
  dataset = tf.data.Dataset.from_tensor_slices(pred_x)
  dataset = dataset.map(_predict_parse_function)
  dataset = dataset.batch(BATCH_SIZE)
  return dataset

# Call the `predict` method to use the trained model to
# make predictions
predictions = model.predict(
    input_fn=predict_input_fn
)

# Print the predictions and display the images
template = ('\n\n\n\nPrediction is "{}" ({:.1f}%) from image:"')
for (prediction, image) in zip(predictions, pred_x):
    class_id = prediction['classes']
    class_name = encoder.inverse_transform([class_id])[0]
    probability = prediction['probabilities'][class_id]
    print(
      template.format(
        class_name,
        100 * probability
      )
    )
    display(Image(image))
