# Databricks notebook source
# MAGIC %md
# MAGIC # Train Krisztina tér with Feature store
# MAGIC
# MAGIC This notebook will train a model to predict bikes at Krisztina tér using Feature store

# COMMAND ----------

import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC Read the transformed data

# COMMAND ----------

gold_table = (spark
  .read
  .table("bence_toth.bubi_spark.gold_table")
  .withColumn("index", F.monotonically_increasing_id())
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Save a table with the bikes at Krisztina körút station

# COMMAND ----------

krisztina_station_id = 2100

# COMMAND ----------

krisztina_table = (gold_table
 .filter(F.col("station_id") == krisztina_station_id)
 .drop("station_id")
)

# COMMAND ----------

krisztina_table.printSchema()

# COMMAND ----------

display(krisztina_table)

# COMMAND ----------

display(krisztina_table.filter(F.col("index") == 17179960684))

# COMMAND ----------

krisztina_table_name = "bubi_spark.krisztina_bikes"


# COMMAND ----------

# MAGIC %md
# MAGIC This DF can be written to a table to allow for AutoML experimentation

# COMMAND ----------

"""
(krisztina_table.write
 .mode("overwrite")
 #.option("overwriteSchema", "true")
 .saveAsTable("bence_toth.bubi_spark.krisztina_bikes")
 )
"""

# COMMAND ----------

# MAGIC %md
# MAGIC # Get a baseline
# MAGIC
# MAGIC What will be the R2, if I predict future bike number simply with the present bike number?

# COMMAND ----------

future_bikes = krisztina_table.select("future_bikes").toPandas()
present_bikes = krisztina_table.select("bikes").toPandas()

from sklearn.metrics import r2_score

r2_baseline = r2_score(future_bikes, present_bikes)

print(f"R2 baseline is: {r2_baseline:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature store

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS hive_metastore.bubi_spark

# COMMAND ----------

from databricks import feature_store

fs = feature_store.FeatureStoreClient()

fs.create_table(
  name=krisztina_table_name,
  schema=krisztina_table.drop("future_bikes").schema,
  primary_keys=["index"],
  description="Bubi bike availability at Krisztina tér station"
)

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE EXTENDED bubi_spark.krisztina_bikes

# COMMAND ----------

fs.write_table(name=krisztina_table_name, df=krisztina_table.drop("future_bikes"), mode="overwrite")

# COMMAND ----------

fs.get_table(krisztina_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC Get data from Feature store and train model

# COMMAND ----------

label_df = krisztina_table.select("index", "future_bikes")

# COMMAND ----------

display(label_df.filter(F.col("index") == 17179960684))

# COMMAND ----------

from databricks.feature_store import FeatureLookup
from sklearn.model_selection import train_test_split

index = "index"
label = "future_bikes"

model_feature_lookups = [FeatureLookup(table_name=krisztina_table_name, lookup_key=index)]

# fs.create_training_set will look up features in model_feature_lookups with matched key from inference_data_df
training_set = fs.create_training_set(label_df, model_feature_lookups, label=label, exclude_columns=index)
training_pd = training_set.load_df().toPandas()

display(training_set.load_df())

# Create train and test datasets
X = training_pd.drop(label, axis=1)
y = training_pd[label]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#X_train.head()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM bubi_spark.krisztina_bikes

# COMMAND ----------

17179960684
8590030235

# COMMAND ----------

training_set.feature_spec

# COMMAND ----------


