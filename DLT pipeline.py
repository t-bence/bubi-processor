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

# MAGIC %md
# MAGIC Run the Utilities notebook to get some helper functions

# COMMAND ----------

# MAGIC %run ./Utilities

# COMMAND ----------

# MAGIC %md
# MAGIC Get the JSON schema defined in the Utilities notebook

# COMMAND ----------


json_schema = get_json_schema()

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
  return (spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
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
  return (dlt.readStream("raw_bubi_data")
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
  return (dlt.readStream("with_timestamp")
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
  return (dlt.readStream("time_deduped")
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
  return (dlt.readStream("row_per_station")
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
  return (dlt.readStream("stations")
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
  return (dlt.readStream("row_per_station")
    .drop("name", "lat", "lng")
  )

# COMMAND ----------


