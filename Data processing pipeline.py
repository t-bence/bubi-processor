# Databricks notebook source
# MAGIC %sql 
# MAGIC USE CATALOG bence_toth;
# MAGIC CREATE SCHEMA IF NOT EXISTS bubi_project;
# MAGIC USE SCHEMA bubi_project;

# COMMAND ----------

import pyspark.sql.functions as F

schema = "bence_toth.bubi_project"

volume_path = "/Volumes/bence_toth/bubi_project/bubi-scraper-v2-volume"

checkpoints_folder = "/tmp/bubi/checkpoints/"

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
# MAGIC
# MAGIC For notifications to work, I would need to set up the following parameters: https://docs.databricks.com/en/ingestion/auto-loader/options.html#azure-specific-options
# MAGIC
# MAGIC I'm skipping that for the sake of simplicity.

# COMMAND ----------

jsons_bronze = (spark
  .readStream
  .format("cloudFiles")
  .option("cloudFiles.format", "json")
  #.option("cloudFiles.useNotifications", "true")
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
  .option("checkpointLocation", checkpoints_folder + "jsons_bronze")
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
  # next 2 lines remove random bikes left around 
  .filter(F.col("places.spot") == F.lit(True)) 
  .filter(F.col("places.bike") == F.lit(False))
  .select("ts", "places")
  # truncate timestamp to minutes only
  .withColumn("ts", F.date_trunc("minute", F.col("ts")))
  # drop first few runs that are sometimes not at a ten-minute interval
  .filter(F.col("ts") >= "2024-03-12T12:00:00.000+00:00")
  # extract the useful columns
  .withColumn("station_id", F.col("places.number"))
  .withColumn("bikes", F.col("places.bikes"))
  .withColumn("maintenance", F.col("places.maintenance"))
  .withColumn("station_name", F.col("places.name"))
  .withColumn("lat", F.col("places.lat"))
  .withColumn("lng", F.col("places.lng"))
  .drop("places")
  # fix timestamps to correspond to always XX:10:00 or XX:20:00 or so
  .transform(timestamp_to_ten_minutes, "ts")
  .dropDuplicates(["ts", "station_id"])
)

# COMMAND ----------

# MAGIC %md
# MAGIC Drop table and checkpoints -- only in development!

# COMMAND ----------

# ONLY IN DEVELOPMENT
#spark.sql("DROP TABLE IF EXISTS snapshots_silver")
#dbutils.fs.rm("dbfs:/tmp/bubi/checkpoints/snapshots_silver", True)

# COMMAND ----------

# MAGIC %md
# MAGIC Save to table

# COMMAND ----------

snapshots_query = (snapshots_silver
  .writeStream
  .outputMode("append")
  .queryName("snapshot_stream")
  .trigger(availableNow=True)
  .option("checkpointLocation", checkpoints_folder + "snapshots_silver")
  .toTable("snapshots_silver")
)

# COMMAND ----------

# MAGIC %md
# MAGIC #stations_silver table
# MAGIC
# MAGIC This should contain the individual stations with the closest five from them. But for now, let's only save the station IDs and positions.
# MAGIC
# MAGIC This is because it can be done with a streaming query.

# COMMAND ----------

stations_query = (spark
  .readStream
  .table("snapshots_silver")
  .select("station_id", "station_name", "lat", "lng")
  .dropDuplicates()
  .withColumn("district", F.substring("station_name", 1, 2).cast("int"))
  .writeStream
  .outputMode("append")
  .queryName("stations_stream")
  .trigger(availableNow=True)
  .option("checkpointLocation", checkpoints_folder + "stations_silver")
  .toTable("stations_silver")
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

stations = spark.read.table("stations_silver")

closest_stations = (stations
  .drop("station_name", "district")
  .crossJoin(stations
    .drop("station_name", "district")
    .withColumnRenamed("station_id", "other_station_id")
    .withColumnRenamed("lat", "other_lat")
    .withColumnRenamed("lng", "other_lng")
  )
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

stations_with_closest = (stations
  .join(closest_stations, "station_id")
  .select("station_id", "closest_stations")
)

# COMMAND ----------

stations_with_closest.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Bike data

# COMMAND ----------

snapshots = (spark
  .readStream
  .table("snapshots_silver")
  .withWatermark("ts", "10 seconds")
)

gold = (snapshots
  .join(stations_with_closest, "station_id", "left")
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
  .withColumn("close_bikes_1", F.element_at("close_bikes", 1))
  .withColumn("close_bikes_2", F.element_at("close_bikes", 2))
  .withColumn("close_bikes_3", F.element_at("close_bikes", 3))
  .withColumn("close_bikes_4", F.element_at("close_bikes", 4))
  .withColumn("close_bikes_5", F.element_at("close_bikes", 5))
  .drop("close_bikes")
)

# COMMAND ----------

# ONLY IN DEVELOPMENT
#spark.sql("DROP TABLE IF EXISTS gold")
#dbutils.fs.rm("dbfs:/tmp/bubi/checkpoints/gold", True)

# COMMAND ----------

(gold
  .writeStream
  .outputMode("append")
  .queryName("gold_stream")
  .trigger(availableNow=True)
  .option("checkpointLocation", checkpoints_folder + "gold")
  .toTable("gold")
)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM gold WHERE close_bikes_5 IS NULL

# COMMAND ----------

# MAGIC %md
# MAGIC # Write data to feature table

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

feature_table_name = schema + ".features"

# COMMAND ----------

# Drop table
#fe.drop_table(name=feature_table_name)

# COMMAND ----------

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
  #checkpoint_location="/tmp/bubi/checkpoints/features", # this cannot be a streaming write
  trigger={"availableNow": True}
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Label creation
# MAGIC
# MAGIC Label: number of bikes at the same station, four hours in the future based on the ts timestamp

# COMMAND ----------

hours_in_future = 4

seconds = hours_in_future * 3600

from pyspark.sql.window import Window

windowSpec = (Window
  .partitionBy("station_id")
  .orderBy(F.unix_timestamp(F.col("ts")))
  .rangeBetween(seconds - 10, seconds + 10)
)

label = (spark.read.table("gold")
  .select("station_id", "ts", "bikes")
  .withColumn("label", F.mean("bikes").over(windowSpec))
  .drop("bikes")
)

label.write.mode("overwrite").saveAsTable(schema + ".label")
