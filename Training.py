# Databricks notebook source
# MAGIC %md
# MAGIC # Training notebook
# MAGIC
# MAGIC This notebook trains ML models for the stations and registers them into Unity Catalog
# MAGIC
# MAGIC Code is based on Pipeline deployment from MLWD 

# COMMAND ----------

# MAGIC %run ./Utilities

# COMMAND ----------

import pyspark.sql.functions as F

schema = "bence_toth.bubi_project"

# COMMAND ----------

# MAGIC %md
# MAGIC The stations IDs to train for

# COMMAND ----------

station_ids = [2100]

# COMMAND ----------

import pyspark.sql.functions as F

schema = "bence_toth.bubi_project"
