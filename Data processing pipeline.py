# Databricks notebook source
# MAGIC %sql 
# MAGIC USE CATALOG bence_toth;
# MAGIC CREATE SCHEMA IF NOT EXISTS bubi_project;
# MAGIC USE SCHEMA bubi_project;

# COMMAND ----------

import pyspark.sql.functions as F

schema = "bence_toth.bubi_project"

volume_path = "/Volumes/bence_toth/bubi_project/bubi-scraper-v2-volume"

# COMMAND ----------

# MAGIC %md
# MAGIC Load the JSON schema from an external notebook

# COMMAND ----------

# MAGIC %run ./Utilities

# COMMAND ----------

# MAGIC %md
# MAGIC Create the raw data, that stores the JSON contents
# MAGIC
# MAGIC Note: input_file_name() does not work in UC, must use the _metadata col instead, which has a field called file_name and one called file_path
# MAGIC
# MAGIC https://community.databricks.com/t5/data-engineering/input-file-name-not-supported-in-unity-catalog/td-p/16042

# COMMAND ----------

# MAGIC %md
# MAGIC # jsons_bronze
# MAGIC This contains the raw jsons and the file names

# COMMAND ----------

jsons_bronze = (spark
  .readStream
  .format("cloudFiles")
  .option("cloudFiles.format", "json")
  .schema(get_json_schema())
  .load(volume_path)
  .withColumn("filename", F.col("_metadata.file_name"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC Save to table

# COMMAND ----------

(jsons_bronze
  .writeStream
  .outputMode("append")
  .queryName("json_stream")
  .trigger(availableNow=True)
  .option("checkpointLocation", "/tmp/delta/events/_checkpoints/")
  .toTable("jsons_bronze")
)

# COMMAND ----------

# MAGIC %md
# MAGIC #snapshots_silver table
# MAGIC This table contains a row for each station at each timestamp

# COMMAND ----------

date_length = len("2024-03-14T01-30-02") # skip trailing Z

snapshots_silver = (jsons_bronze
  .withColumn("ts", F.to_timestamp(
    F.substring("filename", 0, date_length), "yyyy-MM-dd'T'HH-mm-ss"))
  .withWatermark("ts", "10 minutes")
  .withColumn("data", F.element_at("countries", 1))
  .withColumn("cities", F.col("data.cities"))
  .withColumn("data", F.element_at("cities", 1))
  .select("ts", "data")
  .withColumn("places", F.col("data.places"))
  .drop("data")
  .withColumn("places", F.explode("places"))
  .filter(F.col("places.spot") == F.lit(True)) # This removes random bikes left around
  .filter(F.col("places.bike") == F.lit(False))
  # deduplicate for every ten minute interval
  .select("ts", "places")
  # truncate timestamp to minutes only
  .withColumn("ts", F.date_trunc("minute", F.col("ts")))
  # drop first run that is not at a ten-minute interval
  .filter(F.col("ts") >= "2024-03-12T11:00:00.000+00:00")
  # extract the useful columns
  .withColumn("bikes", F.col("places.bikes"))
  .withColumn("maintenance", F.col("places.maintenance"))
  .withColumn("station_name", F.col("places.name"))
  .withColumn("lat", F.col("places.lat"))
  .withColumn("lng", F.col("places.lng"))
  .withColumn("station_id", F.col("places.number"))
  .drop("places")
  .withColumn("date", F.col("ts").cast("date"))
  .withColumn("hour", F.hour("ts"))
  .withColumn("ten_minute", (F.minute("ts") / 10).cast("int"))
  .dropDuplicates(["station_id", "date", "hour", "ten_minute"])
  .drop("date", "hour", "ten_minute")
)

# COMMAND ----------

# MAGIC %md
# MAGIC Save to table

# COMMAND ----------

snapshots_query = (snapshots_silver
  .writeStream
  .outputMode("append")
  .queryName("snapshot_stream")
  .trigger(availableNow=True)
  .option("checkpointLocation", "/tmp/checkpoints")
  .toTable("snapshots_silver")
)

# COMMAND ----------

# MAGIC %md
# MAGIC #stations_silver table
# MAGIC
# MAGIC This should contain the individual stations with the closest five from them

# COMMAND ----------

stations = (spark
  .read
  .table("snapshots_silver")
  .select("station_id", "station_name", "lat", "lng")
  .dropDuplicates()
  .withColumn("district", F.substring("station_name", 1, 2).cast("int"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC Compute distances using Haversine formula

# COMMAND ----------

def dist(long_x, lat_x, long_y, lat_y):
    return F.acos(
        F.sin(F.radians(lat_x)) * F.sin(F.radians(lat_y)) + 
        F.cos(F.radians(lat_x)) * F.cos(F.radians(lat_y)) * 
            F.cos(F.radians(long_x) - F.radians(long_y))
    ) * F.lit(6371.0 * 1000)

# COMMAND ----------

from pyspark.sql import Window

N_closest_stations = 5

distanceWindowSpec = (Window
  .partitionBy("station_id")
  .orderBy(F.col("distance_meters"))
)

closest_stations = (stations
  .drop("station_name", "district")
  .crossJoin(stations
    .drop("station_name", "district")
    .withColumnRenamed("station_id", "other_station_id")
    .withColumnRenamed("lat", "other_lat")
    .withColumnRenamed("lng", "other_lng")
  )
  #.filter(F.col("station_id") < F.col("other_station_id"))
  .filter(F.col("station_id") != F.col("other_station_id"))
  .withColumn("distance_meters",
    dist(F.col("lng"), F.col("lat"), F.col("other_lng"), F.col("other_lat")).cast("int"))
  .select("station_id", "other_station_id", "distance_meters")
  .withColumn("rank", F.dense_rank().over(distanceWindowSpec))
  .filter(F.col("rank") <= N_closest_stations)
  .groupBy("station_id")
  .agg(F.collect_list("other_station_id").alias("closest_stations"))
  # dense_rank give ties, so we have to filter those to have exactly just five
  .withColumn("closest_stations", F.slice("closest_stations", 1, N_closest_stations))
)

# COMMAND ----------

# MAGIC %md
# MAGIC Join closest stations onto stations and write to a batch table

# COMMAND ----------

stations.display()

# COMMAND ----------

(stations
  .join(closest_stations, "station_id")
  .write
  .mode("overwrite")
  .saveAsTable("stations_silver")
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Bike data

# COMMAND ----------

snapshots = (spark
  .readStream
  .table("snapshots_silver")
  .withWatermark("ts", "10 seconds")
)

from pyspark.sql.window import Window

seconds_in_hours = 3600

windowSpec = (Window
  .partitionBy("station_id")
  .orderBy(F.unix_timestamp(F.col("timestamp")))
  .rangeBetween(4 * seconds_in_hours - 10, 4 * seconds_in_hours + 10) # data four hours from now
)

gold = (snapshots
  .join(spark.read.table("stations_silver").select("station_id", "closest_stations"), "station_id", "left")
  .withColumn("close_station", F.explode("closest_stations"))
  .drop("closest_stations")
  .join(snapshots
        .withColumnRenamed("station_id", "close_station")
        .withColumnRenamed("bikes", "close_bikes"),
    ["close_station", "ts"], "left")
  .na.fill(0, "close_bikes")
  .groupBy("station_id", "ts")
  .agg(
    F.first("bikes").alias("bikes"),
    F.collect_list("close_bikes").alias("close_bikes")
  )
  .withColumn("weekday", F.dayofweek("ts"))
  .withColumn("hour", F.hour("ts"))
  .withColumn("tenminute", F.minute("ts") / 10)
  #.withColumn("future_bikes", F.mean("bikes").over(windowSpec))
  .withColumn("close_bikes_1", F.element_at("close_bikes", 1))
  .withColumn("close_bikes_2", F.element_at("close_bikes", 2))
  .withColumn("close_bikes_3", F.element_at("close_bikes", 3))
  .withColumn("close_bikes_4", F.element_at("close_bikes", 4))
  .withColumn("close_bikes_5", F.element_at("close_bikes", 5))
  .drop("close_bikes")
  #.filter(F.col("future_bikes").isNotNull())
)

# COMMAND ----------

(gold
  .writeStream
  .outputMode("append")
  .queryName("gold_stream")
  .trigger(availableNow=True)
  .option("checkpointLocation", "/tmp/bubi/checkpoints/gold")
  .toTable("gold")
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Write data to feature table

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

feature_table_name = schema + ".features"

fe.create_table(name=feature_table_name,
  description="Features for Bubi learning",
  schema=gold.schema,
  timeseries_columns="ts",
  primary_keys=["station_id", "ts"]
)

# COMMAND ----------

fe.write_table(name=feature_table_name,
  mode="merge",
  df=spark.read.table("gold"),
  #checkpoint_location="/tmp/bubi/checkpoints/features",
  trigger={"availableNow": True}
)

# COMMAND ----------


