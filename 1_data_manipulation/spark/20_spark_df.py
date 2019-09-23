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

# ## Spark DataFrame API example (with PySpark)

# Import the required modules
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, count, mean

# Start a Spark session
spark = SparkSession.builder.master('local').getOrCreate()

# Load the data into Spark
flights = spark.table('flights')

# Display a subset of rows from the Spark DataFrame
flights.show()

# Use Spark DataFrame methods to perform operations on the
# DataFrame and return a pointer to the result DataFrame
result = flights \
  .filter(col('dest') == lit('LAS')) \
  .groupBy('origin') \
  .agg( \
       count('*').alias('num_departures'), \
       mean('dep_delay').alias('avg_dep_delay') \
  ) \
  .orderBy('avg_dep_delay')

# Display the result
result.show()

# In this case, the _full_ result Spark DataFrame is
# printed to the screen because it's so small

# You can also assign the result as a pandas DataFrame 
# by calling the `toPandas` method
result.toPandas()

# End the Spark session
spark.stop()
