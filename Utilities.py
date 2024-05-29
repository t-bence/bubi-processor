# Databricks notebook source
from pyspark.sql.types import StructType, StructField, ArrayType, BooleanType, LongType, StringType, DoubleType

# COMMAND ----------

# MAGIC %md
# MAGIC # Input JSON schema

# COMMAND ----------



def get_json_schema():
  return StructType([StructField('countries', ArrayType(StructType([StructField('available_bikes', LongType(), True), StructField('booked_bikes', LongType(), True), StructField('capped_available_bikes', BooleanType(), True), StructField('cities', ArrayType(StructType([StructField('alias', StringType(), True), StructField('available_bikes', LongType(), True), StructField('bike_types', StructType([StructField('150', LongType(), True), StructField('297', LongType(), True), StructField('undefined', LongType(), True)]), True), StructField('booked_bikes', LongType(), True), StructField('bounds', StructType([StructField('north_east', StructType([StructField('lat', DoubleType(), True), StructField('lng', DoubleType(), True)]), True), StructField('south_west', StructType([StructField('lat', DoubleType(), True), StructField('lng', DoubleType(), True)]), True)]), True), StructField('break', BooleanType(), True), StructField('lat', DoubleType(), True), StructField('lng', DoubleType(), True), StructField('maps_icon', StringType(), True), StructField('name', StringType(), True), StructField('num_places', LongType(), True), StructField('places', ArrayType(StructType([StructField('address', StringType(), True), StructField('bike', BooleanType(), True), StructField('bike_list', ArrayType(StructType([StructField('active', BooleanType(), True), StructField('battery_pack', StringType(), True), StructField('bike_type', LongType(), True), StructField('boardcomputer', LongType(), True), StructField('electric_lock', BooleanType(), True), StructField('lock_types', ArrayType(StringType(), True), True), StructField('number', StringType(), True), StructField('pedelec_battery', StringType(), True), StructField('state', StringType(), True)]), True), True), StructField('bike_numbers', ArrayType(StringType(), True), True), StructField('bike_racks', LongType(), True), StructField('bike_types', StructType([StructField('150', LongType(), True), StructField('297', LongType(), True), StructField('undefined', LongType(), True)]), True), StructField('bikes', LongType(), True), StructField('bikes_available_to_rent', LongType(), True), StructField('booked_bikes', LongType(), True), StructField('free_racks', LongType(), True), StructField('free_special_racks', LongType(), True), StructField('lat', DoubleType(), True), StructField('lng', DoubleType(), True), StructField('maintenance', BooleanType(), True), StructField('name', StringType(), True), StructField('number', LongType(), True), StructField('place_type', StringType(), True), StructField('rack_locks', BooleanType(), True), StructField('special_racks', LongType(), True), StructField('spot', BooleanType(), True), StructField('terminal_type', StringType(), True), StructField('uid', LongType(), True)]), True), True), StructField('refresh_rate', StringType(), True), StructField('return_to_official_only', BooleanType(), True), StructField('set_point_bikes', LongType(), True), StructField('uid', LongType(), True), StructField('website', StringType(), True), StructField('zoom', LongType(), True)]), True), True), StructField('country', StringType(), True), StructField('country_calling_code', StringType(), True), StructField('country_name', StringType(), True), StructField('currency', StringType(), True), StructField('domain', StringType(), True), StructField('email', StringType(), True), StructField('faq_url', StringType(), True), StructField('hotline', StringType(), True), StructField('language', StringType(), True), StructField('lat', DoubleType(), True), StructField('lng', DoubleType(), True), StructField('name', StringType(), True), StructField('no_registration', BooleanType(), True), StructField('policy', StringType(), True), StructField('pricing', StringType(), True), StructField('set_point_bikes', LongType(), True), StructField('show_bike_type_groups', BooleanType(), True), StructField('show_bike_types', BooleanType(), True), StructField('show_free_racks', BooleanType(), True), StructField('store_uri_android', StringType(), True), StructField('store_uri_ios', StringType(), True), StructField('system_operator_address', StringType(), True), StructField('terms', StringType(), True), StructField('timezone', StringType(), True), StructField('vat', StringType(), True), StructField('website', StringType(), True), StructField('zoom', LongType(), True)]), True), True)])


# COMMAND ----------

# MAGIC %md
# MAGIC # Timestamp rounding
# MAGIC
# MAGIC Timestamp rounding function -- rounds the timestamp to the nearest 10 minutes, in a floor fashion. E.g.:
# MAGIC
# MAGIC `2024-03-12 12:11:50 -> 2024-03-12 12:10:00`

# COMMAND ----------

from pyspark.sql.dataframe import DataFrame
import pyspark.sql.functions as F

def timestamp_to_ten_minutes(input: DataFrame, column: str) -> DataFrame:
  return (input
    .withColumn(column, F.to_timestamp(F.col(column).cast("long") - (F.minute(column) % 10) * 60 - F.second(column)))
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Unit tests for the function:

# COMMAND ----------

from datetime import datetime

timestamps = [
    datetime(2024, 4, 25, 10, 30, 1),
    datetime(2024, 4, 25, 11, 31, 2),
    datetime(2024, 4, 25, 13, 39, 59),
    datetime(2024, 4, 25, 14, 0, 0),
    datetime(2024, 4, 25, 15, 0, 5)
]

# Create a DataFrame with timestamps
df = spark.createDataFrame([(ts,) for ts in timestamps], ["timestamp"])

result = df.transform(timestamp_to_ten_minutes, "timestamp")

timestamps = [x.timestamp for x in result.collect()]

# COMMAND ----------

assert timestamps[0] == datetime(2024, 4, 25, 10, 30, 0)
assert timestamps[1] == datetime(2024, 4, 25, 11, 30, 0)
assert timestamps[2] == datetime(2024, 4, 25, 13, 30, 0)
assert timestamps[3] == datetime(2024, 4, 25, 14, 0, 0)
assert timestamps[4] == datetime(2024, 4, 25, 15, 0, 0)
print("Tests passed")

# COMMAND ----------

# MAGIC %md
# MAGIC # Get latest model version from UC

# COMMAND ----------

def get_latest_model_version(model_name: str):
  """Helper function to get latest model version"""
  import mlflow
  mlflow.set_registry_uri("databricks-uc")

  client = mlflow.MlflowClient()

  model_version_infos = client.search_model_versions(f"name = '{model_name}'")
  return max([model_version_info.version for model_version_info in model_version_infos])

# COMMAND ----------


