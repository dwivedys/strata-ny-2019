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

# # Select specific columns from a pandas DataFrame

# Import modules and read games data
import numpy as np
import pandas as pd
games = pd.read_table('data/games/games.csv', sep=',')
games


# Use the DataFrame method `filter` to return a DataFrame 
# containing only some of the columns from the `games` 
# DataFrame. Specify a list of quoted names of the 
# columns to return
games.filter(['name', 'min_players', 'max_players'])


# Write the expression on multiple lines
games \
  .filter(['name', 'min_players', 'max_players'])

# or

(games
  .filter(['name', 'min_players', 'max_players']))


# Alternatively, use the `.loc` indexer:
games \
  .loc[:, ['name', 'min_players', 'max_players']]


# To return a single column as a `pandas.Series`,
# use `.loc` with a single column name not enclosed
# in a list
games.loc[:, 'name']

# Alternatively, you can return a single column as a 
# Series by using square brackets or dot syntax
games['name']
games.name

# It is convenient but unsafe to use the dot syntax.
# If the column name conflicts with the name of a 
# `pandas.Series` method or attribute, then 
# the dot syntax will return unexpected results.
