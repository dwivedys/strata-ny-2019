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

# # Select specific columns from an R data frame using dplyr

# Load packages and read games data
library(readr)
games <- read_csv("data/games/games.csv")
games


# Load the dplyr package
library(dplyr)

# Use the dplyr verb `select()` to return a data frame
# (tibble) containing only some of the columns from the 
# `games` data frame
games %>% select(name, min_players, max_players)

# Write the expression on multiple lines
games %>% 
  select(name, min_players, max_players)


# `select()` always returns a data frame, even if
# it only contains one column
games %>% select(name)

# To return a single column as a vector, use the
# dplyr function `pull()`
games %>% pull(name)

# Alternatively, you can return a vector by using 
# double square brackets or the dollar-sign operator
games[["name"]]
games$name
