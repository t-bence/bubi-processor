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

(jsons_bronze
  .writeStream
  .outputMode("append")
  .queryName("bikes_stream")
  .trigger(availableNow=True)
  .option("checkpointLocation", "/tmp/delta/events/_checkpoints/")
  .toTable("jsons_bronze")
)

# COMMAND ----------

# MAGIC %md
# MAGIC #snapshots_silver table
# MAGIC This table contains a row for each station at each timestamp

# COMMAND ----------

snapshots_silver = (jsons_bronze
  .withColumn("ts", F.to_timestamp(
    F.substring("filename", 0, -1), "yyyy-MM-dd'T'HH-mm-ss"))
)

# COMMAND ----------


