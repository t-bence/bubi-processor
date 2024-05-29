# Databricks notebook source
# MAGIC %md
# MAGIC This notebook will do inference on the new, incoming data and store it to a table.
# MAGIC
# MAGIC Furthermore, it will create a table that returns the most recent predictions only.

# COMMAND ----------

import pyspark.sql.functions as F

schema = "bence_toth.bubi_project"

# COMMAND ----------

model_name = schema + ".model"

# COMMAND ----------

# MAGIC %md
# MAGIC This part is based on [the Scalable ML course](https://adb-3679152566148441.1.azuredatabricks.net/?o=3679152566148441#notebook/4193177320851625/command/4193177320851646)

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

# Feature table definition
fe = FeatureEngineeringClient()

# Load the model
model_uri = f"models:/{model_name}@delegating"

# COMMAND ----------

labels = spark.read.table(schema + ".label")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC Do the actual inference. This should be a streaming job later on, but for now it is a batch one.

# COMMAND ----------

result_df = fe.score_batch(
    model_uri=model_uri,
    df=labels,
    result_type="int"
)

# COMMAND ----------

result_df.show()

# COMMAND ----------

#result_df.write.saveAsTable(schema + ".predictions")

# COMMAND ----------

# MAGIC %md
# MAGIC Write the newest, most current results into another table that can be queried

# COMMAND ----------

most_current_timestamp = result_df.select(F.max("ts")).collect()[0][0]

current_predictions = (result_df
  .filter(F.col("ts") == most_current_timestamp)
  .display()
  #.write
  #.mode("overwrite")
  #.saveAsTable(schema + ".current_predictions")
)
