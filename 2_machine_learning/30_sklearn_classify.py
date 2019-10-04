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

# # Decision tree classifier with scikit-learn

# This example demonstrates a decision tree classification
# task using the using the
# [`DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
# class in the 
# [`sklearn.tree`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree)
# module.


# ## 0. Preliminaries

# Import the required modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


# ## 1. Load data

# Load data representing one brand of chess set ("set A")
chess = pd.read_csv('data/chess/one_chess_set.csv')

# View the data
chess


# ## 2. Prepare data

# Separate the features (x) and labels (y)
features = chess.filter(['base_diameter','height'])
labels = chess.filter(['piece'])

# Encode the character string labels as integer
# numbers using the
# [`LabelEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
# class in the 
# [`sklearn.preprocessing`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing)
# module
label_encoder = LabelEncoder()
label_encoder.fit(labels.piece)
labels = label_encoder.transform(labels.piece)

# Split the training and test data each into an 80% 
# training set and a 20% test set using 
# scikit-learn's 
# [`train_test_split`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
# function
train_x, test_x, train_y, test_y = train_test_split(
  features,
  labels,
  test_size=0.2
)


# ## 3. Specify model

# Create the decision tree classifier model object
# ("estimator") by calling the
# `DecisionTreeClassifier` function
model = DecisionTreeClassifier()


# ## 4. Train model

# Call the `fit` method to train the model
model.fit(train_x, train_y)


# ## 5. Evaluate model

# Call the `score` method to compute the accuracy
# on the test set. This is the proportion of 
# records in the test set whose labels were correctly
# predicted by the model
model.score(test_x, test_y)


# ## 6(b). Make predictions

# See what predictions the trained model generates for
# six new rows of data (feature only)
d = {
     'base_diameter': [27.3, 32.7, 31, 32.1, 35.9, 37.4],
     'height': [45.7, 58.1, 65.2, 46.3, 75.6, 95.4]
   }
new_data = pd.DataFrame(data=d)

# Call the `predict` method to use the trained model to
# make predictions on this new data
predictions = model.predict(new_data)

# Print the predictions
print(predictions)

label_encoder.inverse_transform(predictions)
