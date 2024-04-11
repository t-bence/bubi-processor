# Databricks notebook source
# MAGIC %md
# MAGIC # Feature engineering
# MAGIC
# MAGIC Let's create the feature table for the ML model. 
# MAGIC
# MAGIC Features:
# MAGIC - number of bikes now at the station
# MAGIC - number of bikes at the closest five stations
# MAGIC - time: hours
# MAGIC - day of week
# MAGIC
# MAGIC Label:
# MAGIC - number of bikes at the station, four hours from now

# COMMAND ----------

import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC Read the stations

# COMMAND ----------

stations = spark.read.table("bence_toth.bubi_spark.stations")

# COMMAND ----------

display(stations)

# COMMAND ----------

# MAGIC %md
# MAGIC How many closest stations are we collecting?

# COMMAND ----------

N_closest_stations = len(stations.take(1)[0].closest_stations)
print(N_closest_stations)

# COMMAND ----------

# MAGIC %md
# MAGIC Read the bike number data

# COMMAND ----------

bikes_at_stations = (spark.read.table("bence_toth.bubi_spark.bikes_at_stations")
    .select("station_id", "timestamp", "bikes")
)

# COMMAND ----------

display(bikes_at_stations)

# COMMAND ----------

bikes_with_closest_data = (bikes_at_stations
  .join(stations.select("station_id", "closest_stations"), "station_id", "left")
  .withColumn("close_station", F.explode("closest_stations"))
  .drop("closest_stations")
  .join(bikes_at_stations
        .withColumnRenamed("station_id", "close_station")
        .withColumnRenamed("bikes", "close_bikes"),
    ["close_station", "timestamp"], "left")
  .na.fill(0, "close_bikes")
  .groupBy("station_id", "timestamp")
  .agg(
    F.first("bikes").alias("bikes"),
    F.collect_list("close_bikes").alias("close_bikes")
  )
)

# COMMAND ----------

display(bikes_with_closest_data)

# COMMAND ----------

bikes_with_closest_data.count()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM bence_toth.bubi_spark.bikes_at_stations WHERE station_id = 2029 AND timestamp = "2024-03-12T11:50:00.000+00:00"

# COMMAND ----------

# MAGIC %md
# MAGIC Here I need to get the label: the number of bikes four hours from now

# COMMAND ----------

bikes_with_closest_data.printSchema()

# COMMAND ----------

from pyspark.sql.window import Window

seconds_in_hours = 3600

windowSpec = (Window
  .partitionBy("station_id")
  .orderBy(F.unix_timestamp(F.col("timestamp")))
  .rangeBetween(4 * seconds_in_hours - 10, 4 * seconds_in_hours + 10) # data four hours from now
)

gold_table = (bikes_with_closest_data
  .withColumn("weekday", F.dayofweek("timestamp"))
  .withColumn("hour", F.hour("timestamp"))
  .withColumn("future_bikes", F.mean("bikes").over(windowSpec))
)

# COMMAND ----------

display(gold_table.orderBy("station_id", "timestamp"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Save a table with the bikes at Krisztina körút station to play with AutoML

# COMMAND ----------

krisztina_table = (gold_table
 .filter(F.col("station_id") == 2100)
 .drop("station_id")
 .withColumn("minute", F.minute("timestamp"))
 .drop("timestamp")
)

# COMMAND ----------

(krisztina_table.write
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable("bence_toth.bubi_spark.krisztina_bikes")
 )

# COMMAND ----------

gold_table.count()

# COMMAND ----------

gold_table.filter(F.col("bikes").isNotNull()).count()

# COMMAND ----------

krisztina_table.printSchema()

# COMMAND ----------

krisztina_table.drop("future_bikes").dropDuplicates().count()

# COMMAND ----------

from databricks import feature_store

fs = feature_store.FeatureStoreClient()

fs.write_table(
  name="bubi_bikes"
)

# COMMAND ----------


