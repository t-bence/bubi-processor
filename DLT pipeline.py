# Databricks notebook source
# MAGIC %md
# MAGIC This notebook contains the DLT pipeline to process the incoming BUBI bike sharing data. The data is read from an Azure container through a UC Volume.

# COMMAND ----------

import dlt
import pyspark.sql.functions as F

volume_path = "/Volumes/bence_toth/bubi_project/bubi-scraper-v2-volume"

# COMMAND ----------

# MAGIC %md
# MAGIC The JSON schema

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, ArrayType, BooleanType, LongType, StringType, DoubleType

json_schema = StructType([StructField('countries', ArrayType(StructType([StructField('available_bikes', LongType(), True), StructField('booked_bikes', LongType(), True), StructField('capped_available_bikes', BooleanType(), True), StructField('cities', ArrayType(StructType([StructField('alias', StringType(), True), StructField('available_bikes', LongType(), True), StructField('bike_types', StructType([StructField('150', LongType(), True), StructField('297', LongType(), True), StructField('undefined', LongType(), True)]), True), StructField('booked_bikes', LongType(), True), StructField('bounds', StructType([StructField('north_east', StructType([StructField('lat', DoubleType(), True), StructField('lng', DoubleType(), True)]), True), StructField('south_west', StructType([StructField('lat', DoubleType(), True), StructField('lng', DoubleType(), True)]), True)]), True), StructField('break', BooleanType(), True), StructField('lat', DoubleType(), True), StructField('lng', DoubleType(), True), StructField('maps_icon', StringType(), True), StructField('name', StringType(), True), StructField('num_places', LongType(), True), StructField('places', ArrayType(StructType([StructField('address', StringType(), True), StructField('bike', BooleanType(), True), StructField('bike_list', ArrayType(StructType([StructField('active', BooleanType(), True), StructField('battery_pack', StringType(), True), StructField('bike_type', LongType(), True), StructField('boardcomputer', LongType(), True), StructField('electric_lock', BooleanType(), True), StructField('lock_types', ArrayType(StringType(), True), True), StructField('number', StringType(), True), StructField('pedelec_battery', StringType(), True), StructField('state', StringType(), True)]), True), True), StructField('bike_numbers', ArrayType(StringType(), True), True), StructField('bike_racks', LongType(), True), StructField('bike_types', StructType([StructField('150', LongType(), True), StructField('297', LongType(), True), StructField('undefined', LongType(), True)]), True), StructField('bikes', LongType(), True), StructField('bikes_available_to_rent', LongType(), True), StructField('booked_bikes', LongType(), True), StructField('free_racks', LongType(), True), StructField('free_special_racks', LongType(), True), StructField('lat', DoubleType(), True), StructField('lng', DoubleType(), True), StructField('maintenance', BooleanType(), True), StructField('name', StringType(), True), StructField('number', LongType(), True), StructField('place_type', StringType(), True), StructField('rack_locks', BooleanType(), True), StructField('special_racks', LongType(), True), StructField('spot', BooleanType(), True), StructField('terminal_type', StringType(), True), StructField('uid', LongType(), True)]), True), True), StructField('refresh_rate', StringType(), True), StructField('return_to_official_only', BooleanType(), True), StructField('set_point_bikes', LongType(), True), StructField('uid', LongType(), True), StructField('website', StringType(), True), StructField('zoom', LongType(), True)]), True), True), StructField('country', StringType(), True), StructField('country_calling_code', StringType(), True), StructField('country_name', StringType(), True), StructField('currency', StringType(), True), StructField('domain', StringType(), True), StructField('email', StringType(), True), StructField('faq_url', StringType(), True), StructField('hotline', StringType(), True), StructField('language', StringType(), True), StructField('lat', DoubleType(), True), StructField('lng', DoubleType(), True), StructField('name', StringType(), True), StructField('no_registration', BooleanType(), True), StructField('policy', StringType(), True), StructField('pricing', StringType(), True), StructField('set_point_bikes', LongType(), True), StructField('show_bike_type_groups', BooleanType(), True), StructField('show_bike_types', BooleanType(), True), StructField('show_free_racks', BooleanType(), True), StructField('store_uri_android', StringType(), True), StructField('store_uri_ios', StringType(), True), StructField('system_operator_address', StringType(), True), StructField('terms', StringType(), True), StructField('timezone', StringType(), True), StructField('vat', StringType(), True), StructField('website', StringType(), True), StructField('zoom', LongType(), True)]), True), True)])


# COMMAND ----------

# MAGIC %md
# MAGIC Create the raw data, that stores the JSON contents
# MAGIC
# MAGIC Note: input_file_name() does not work in UC, must use the _metadata col instead, which has a field called file_name and one called file_path
# MAGIC
# MAGIC https://community.databricks.com/t5/data-engineering/input-file-name-not-supported-in-unity-catalog/td-p/16042

# COMMAND ----------

@dlt.table(
  comment="All incoming data from the Bubi JSON",
  table_properties = {"quality": "bronze"}
)
def raw_bubi_data():
  return (spark.read
    .format("json")
    .schema(json_schema)
    .load(volume_path)
    .withColumn("filename", F.col("_metadata.file_name"))
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Let's get the timestamp from the filename string and transform the JSON to get the first (and only) country data from it
# MAGIC
# MAGIC

# COMMAND ----------

date_length = len("2024-03-14T01-30-02") # skip trailing Z

@dlt.table(
  comment="Data with timestamps and some JSON transforms",
  table_properties = {"quality": "silver"}
)
def with_timestamp():
  return (dlt.read("raw_bubi_data")
    .withColumn("date", F.substring("filename", 0, date_length))
    .withColumn("timestamp", F.to_timestamp("date", "yyyy-MM-dd'T'HH-mm-ss"))
    .select("countries", "timestamp")
    .withColumn("data", F.element_at("countries", 1))
    .withColumn("cities", F.col("data.cities"))
    .withColumn("data", F.element_at("cities", 1))
    .select("timestamp", "data")
    .withColumn("places", F.col("data.places"))
    .drop("data")
  )


# COMMAND ----------

# MAGIC %md
# MAGIC Deduplicate the data so that in contains always just one entry per every ten minute interval

# COMMAND ----------

from pyspark.sql.window import Window

windowSpec = (Window
  .partitionBy("date", "hour", "ten_minute")
  .orderBy(F.col("timestamp"))
)

@dlt.table(
  comment="Deduplicated to contain one record per ten minutes",
  table_properties = {"quality": "silver"}
)
def time_deduped():
  return (dlt.read("with_timestamp")
    .withColumn("date", F.col("timestamp").cast("date"))
    .withColumn("hour", F.hour("timestamp"))
    .withColumn("ten_minute", (F.minute("timestamp") / 10).cast("int"))
    .withColumn("measurement_in_ten_mins", F.dense_rank().over(windowSpec))
    .filter(F.col("measurement_in_ten_mins") == 1)
    .drop("measurement_in_ten_mins")
    .withColumn("minute", F.col("ten_minute") * 10)
    .drop("ten_minute")
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Further transform it to have one record per station. The JSON contains randomly left around bikes, too, these are filtered out here

# COMMAND ----------

@dlt.table(
  comment="Contains one row per station per ten minutes",
  table_properties = {"quality": "silver"}
)
def row_per_station():
  return (dlt.read("time_deduped")
    .withColumn("places", F.explode("places"))
    .filter(F.col("places.spot") == F.lit(True)) # This removes random bikes left around
    .filter(F.col("places.bike") == F.lit(False))
    .withColumn("bikes", F.col("places.bikes"))
    .withColumn("maintenance", F.col("places.maintenance"))
    .withColumn("name", F.col("places.name"))
    .withColumn("lat", F.col("places.lat"))
    .withColumn("lng", F.col("places.lng"))
    .withColumn("station_id", F.col("places.number"))
    .drop("places")
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Let's get the stations

# COMMAND ----------

@dlt.table(
  comment="The BUBI stations",
  table_properties = {"quality": "gold"}
)
def stations():
  return (dlt.read("row_per_station")
  .select("station_id", "name", "lat", "lng")
  .dropDuplicates()
  .withColumn("district", F.substring("name", 1, 2).cast("int"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC Compute distances between the stations. For this, let's have the Haversine formula
# MAGIC
# MAGIC https://stackoverflow.com/questions/38994903/how-to-sum-distances-between-data-points-in-a-dataset-using-pyspark

# COMMAND ----------

def dist(long_x, lat_x, long_y, lat_y):
    return F.acos(
        F.sin(F.radians(lat_x)) * F.sin(F.radians(lat_y)) + 
        F.cos(F.radians(lat_x)) * F.cos(F.radians(lat_y)) * 
            F.cos(F.radians(long_x) - F.radians(long_y))
    ) * F.lit(6371.0 * 1000)

# COMMAND ----------

@dlt.table(
  comment="Distances between BUBI stations, from lower ID to higher",
  table_properties = {"quality": "gold"}
)
def station_distances():
  return (dlt.read("stations")
    .drop("name", "district")
    .crossJoin(dlt.read("stations")
      .drop("name", "district")
      .withColumnRenamed("station_id", "other_station_id")
      .withColumnRenamed("lat", "other_lat")
      .withColumnRenamed("lng", "other_lng")
    )
    .filter(F.col("station_id") < F.col("other_station_id"))
    .withColumn("distance_meters", dist(F.col("lng"), F.col("lat"), F.col("other_lng"), F.col("other_lat")).cast("int"))
    .select("station_id", "other_station_id", "distance_meters")
  )

# COMMAND ----------

@dlt.table(
  comment="Bike numbers per station per ten minutes",
  table_properties = {"quality": "gold"}
)
def bikes_at_stations():
  return (dlt.read("row_per_station")
    .drop("name", "lat", "lng")
  )

# COMMAND ----------


