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

# # Linear regression model with parsnip

# This example demonstrates a simple regression modeling
# task using the
# [parsnip](https://tidymodels.github.io/parsnip/)
# package, with help from two other packages in the
# [tidymodels](https://github.com/tidymodels) collection
# of packages


# ## 0. Preliminaries

# Load the required packages
library(readr)
library(dplyr)
library(rsample)
library(parsnip)
library(yardstick)


# ## 1. Load data

# Load data representing one brand of chess set ("set A")
chess <- read_csv("data/chess/one_chess_set.csv")


# ## 2. Prepare data

# Split the data into an 80% training set and a 20%
# evaluation (test) set, using the `initial_split()`
# function in the
# [rsample](https://tidymodels.github.io/rsample/)
# package
chess_split <- initial_split(chess, prop = 0.8)
chess_train <- training(chess_split)
chess_test <- testing(chess_split)


# ## 3 and 4. Specify and train model

# To train a model with parsnip, you use one of the
# **model** functions listed in the 
# [list of models](https://tidymodels.github.io/parsnip/articles/articles/Models.html)
# on the parsnip website. You specify an _engine_
# (typically an R modeling package) using the 
# `set_engine()` function, then you call the
# `fit()` function to train the model.

model <- linear_reg() %>%
  set_engine("lm") %>%
  fit(weight ~ base_diameter, data = chess_train)


# ## 5. Evaluate model

# To evaluate the model, first you use the model to 
# generate predictions for the test (evaulation) set

# To generate predictions from the trained model, call
# parsnip's `predict()` function, passing the trained model
# object as the first argument, and the data to predict
# on as the `new_data` argument
test_pred <- predict(model, new_data = chess_test)

# Then combine the column of predictions with the column
# of actual target values, so these two columns are
# together in one data frame
test_results <- bind_cols(
  test_pred,
  chess_test %>% select(weight)
)

# Then to evaluate the model, use the `metrics()`
# function in the
# [yardstick](https://tidymodels.github.io/yardstick/)
# package. This function estimates several common
# model performance metrics
test_results %>% 
  metrics(truth = weight, estimate = .pred) 


# ## 6(a). Interpret the model

# Print the coefficient (slope) and intercept of the
# linear regression model
model$fit$coefficients


# ## 6(b). Make predictions

# See what predictions the trained model generates for
# six new rows of data (predictor variables only)
new_data <- tibble(
  base_diameter = c(27.3, 32.7, 31, 32.1, 35.9, 37.4),
  height = c(45.7, 58.1, 65.2, 46.3, 75.6, 95.4)
)

# Call the `predict` function to use the trained model to
# make predictions on this new data
predictions = predict(model, new_data)

# Print the predictions
predictions


# # Other models with parsnip

# To see the list of all models that can be trained
# using the parsnip package, see the 
# [list of models](https://tidymodels.github.io/parsnip/articles/articles/Models.html)
# on the parsnip website.


# # Hyperparameter tuning

# The
# [dials](https://tidymodels.github.io/dials/) package
# provides tools for tuning hyperparameters of models
# that were trained using the parsnip package.
