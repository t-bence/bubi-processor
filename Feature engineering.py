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

#display(bikes_with_closest_data)

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

# MAGIC %md
# MAGIC Create gold table for learning
# MAGIC
# MAGIC Array type is not suitable, so I will split the close_bike column. ML training works, but the KernelExplainer fails and is not able to explain feature importances when I use a vector.

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
  .withColumn("minute", F.minute("timestamp"))
  .withColumn("future_bikes", F.mean("bikes").over(windowSpec))
  .drop("timestamp")
  .withColumn("close_bikes_1", F.element_at("close_bikes", 1))
  .withColumn("close_bikes_2", F.element_at("close_bikes", 2))
  .withColumn("close_bikes_3", F.element_at("close_bikes", 3))
  .withColumn("close_bikes_4", F.element_at("close_bikes", 4))
  .withColumn("close_bikes_5", F.element_at("close_bikes", 5))
  .drop("close_bikes")
  .filter(F.col("future_bikes").isNotNull())
  
)

# COMMAND ----------

display(gold_table.orderBy("station_id"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Write to feature store here! Add the ID already before writing! 
# MAGIC
# MAGIC What do we do with the label? 
# MAGIC Füli, Petz Tomi, Litter Ádám, Endes Peti
# MAGIC
# MAGIC El kellene menteni a labelt is az indexel együtt, de az nem egy másik pipeline-ban kellene legyen, hogy itt tudjunk majd új adattal is dolgozni, amire még nincs label?

# COMMAND ----------

gold_table.write.mode("overwrite").saveAsTable("bence_toth.bubi_spark.gold_table")

# COMMAND ----------


