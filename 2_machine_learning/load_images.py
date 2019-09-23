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

import os
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img

def load_labeled_images(img_root, chess_sets):
  # Specify the unique labels (names of the chess pieces)
  chess_pieces = ['King', 'Queen', 'Rook', 'Bishop', 'Knight', 'Pawn']

  # Make empty lists to hold image file paths and labels
  (paths, labels) = ([], [])

  # Fill the empty lists with the file paths and labels
  for chess_set in chess_sets:
    for chess_piece in chess_pieces:
      img_dir = img_root + '/' + chess_set + '/' + chess_piece + '/'
      img_paths = [img_dir + d for d in os.listdir(img_dir)]
      img_labels = [chess_piece] * len(img_paths)
      paths.extend(img_paths)
      labels.extend(img_labels)

  # Load the images into a list
  images = []  
  for path in paths:
    images.append(load_img(path, color_mode='grayscale'))

  # Represent the images as arrays of numbers
  features = []
  for image in images:
    features.append(img_to_array(image))

  # Return the features and labels as NumPy arrays
  features = np.asarray(features)
  labels = np.asarray(labels)
  return (features, labels)


def load_unlabeled_images(img_root, chess_sets):
  # Make empty list to hold image file paths
  paths = []

  # Fill the empty lists with the file paths and labels
  for chess_set in chess_sets:
    img_dir = img_root + '/' + chess_set + '/'
    img_paths = [img_dir + d for d in os.listdir(img_dir)]
    paths.extend(img_paths)

  # Load the images into a list
  images = []  
  for path in paths:
    images.append(load_img(path, color_mode='grayscale'))

  # Represent the images as arrays of numbers
  features = []
  for image in images:
    features.append(img_to_array(image))

  # Return the features as a NumPy array
  features = np.asarray(features)
  return features
