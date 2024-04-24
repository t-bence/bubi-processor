# Databricks notebook source
# MAGIC %md
# MAGIC # Spark pipeline
# MAGIC
# MAGIC This pipeline used to be equivalent to the DLT pipeline but stores tables instead of materialized views. This is to avoid issues with materialized views, that can only be queried from Shared clusters, which in turn do not support ML runtimes.
# MAGIC
# MAGIC Recently, it has been modified, so now it is different than the DLT one.
# MAGIC
# MAGIC This is a batch processing notebook which could be upgraded with AutoLoader and streaming.

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG bence_toth

# COMMAND ----------

# MAGIC %md
# MAGIC Create a schema to store our results

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS bubi_spark;
# MAGIC USE SCHEMA bubi_spark;

# COMMAND ----------

schema = "bence_toth.bubi_spark"

# COMMAND ----------

import pyspark.sql.functions as F

volume_path = "/Volumes/bence_toth/bubi_project/bubi-scraper-v2-volume"

# COMMAND ----------

# MAGIC %md
# MAGIC The JSON schema

# COMMAND ----------

# MAGIC %run ./Utilities

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

raw_bubi_data = (spark.read
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

with_timestamp = (raw_bubi_data
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

time_deduped = (with_timestamp
  .withColumn("date", F.col("timestamp").cast("date"))
  .withColumn("hour", F.hour("timestamp"))
  .withColumn("ten_minute", (F.minute("timestamp") / 10).cast("int"))
  .withColumn("measurement_in_ten_mins", F.dense_rank().over(windowSpec))
  .filter(F.col("measurement_in_ten_mins") == 1)
  .drop("measurement_in_ten_mins")
  .withColumn("minute", F.col("ten_minute") * 10)
  .drop("ten_minute")
  # truncate timestamp to minutes only
  .withColumn("timestamp", F.date_trunc("minute", F.col("timestamp")))
  # drop first run that is not at a ten-minute interval
  .filter(F.col("timestamp") >= "2024-03-12T11:00:00.000+00:00")
)

# COMMAND ----------

display(time_deduped.orderBy("timestamp"))

# COMMAND ----------

# MAGIC %md
# MAGIC Further transform it to have one record per station. The JSON contains randomly left around bikes, too, these are filtered out here

# COMMAND ----------

row_per_station = (time_deduped
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

stations = (row_per_station
  .select("station_id", "name", "lat", "lng")
  .dropDuplicates()
  .withColumn("district", F.substring("name", 1, 2).cast("int"))
)

stations.cache().count()

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

station_distances = (stations
  .drop("name", "district")
  .crossJoin(stations
    .drop("name", "district")
    .withColumnRenamed("station_id", "other_station_id")
    .withColumnRenamed("lat", "other_lat")
    .withColumnRenamed("lng", "other_lng")
  )
  #.filter(F.col("station_id") < F.col("other_station_id"))
  .filter(F.col("station_id") != F.col("other_station_id"))
  .withColumn("distance_meters", dist(F.col("lng"), F.col("lat"), F.col("other_lng"), F.col("other_lat")).cast("int"))
  .select("station_id", "other_station_id", "distance_meters")
)

# COMMAND ----------

# MAGIC %md 
# MAGIC Finalize the fact table

# COMMAND ----------

bikes_at_stations = (row_per_station
  .drop("name", "lat", "lng")
)

# COMMAND ----------

# MAGIC %md
# MAGIC Add the closest N station IDs to the list of stations

# COMMAND ----------

N_closest_stations = 5

distanceWindowSpec = (Window
  .partitionBy("station_id")
  .orderBy(F.col("distance_meters"))
)

closest_stations = (station_distances
  .withColumn("rank", F.dense_rank().over(distanceWindowSpec))
  .filter(F.col("rank") <= N_closest_stations)
  .groupBy("station_id")
  .agg(F.collect_list("other_station_id").alias("closest_stations"))
  # dense_rank give ties, so we have to filter those to have exactly just five
  .withColumn("closest_stations", F.slice("closest_stations", 1, N_closest_stations))
)

# COMMAND ----------

display(closest_stations)

# COMMAND ----------

# MAGIC %md
# MAGIC This is to find problems because of ties -- should not find any rows now

# COMMAND ----------

(closest_stations
  .withColumn("length", F.array_size("closest_stations"))
  .filter(F.col("length") != N_closest_stations)
  .count()
)

# COMMAND ----------

stations = (stations
  .join(closest_stations, "station_id")
)

# COMMAND ----------

display(stations)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Let's write the silver tables

# COMMAND ----------

bikes_at_stations.write.mode("overwrite").saveAsTable(schema + ".bikes_at_stations")

# COMMAND ----------

stations.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(schema + ".stations")

# COMMAND ----------

station_distances.write.mode("overwrite").saveAsTable(schema + ".station_distances")
