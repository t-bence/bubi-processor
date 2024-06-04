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

# COMMAND ----------

labels = spark.read.table(schema + ".label")

# COMMAND ----------

# MAGIC %md
# MAGIC Here we load data directly from the feature table. This is not nice, could be improved to use Feature store, but it is not 100% clear how to do it in a streaming way.

# COMMAND ----------

features = spark.read.table(schema + ".features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC Do the actual inference. This should be a streaming job later on, but for now it is a batch one.

# COMMAND ----------

apply_return_schema = "station_id int, ts timestamp, prediction int"

import pandas as pd
import mlflow
mlflow.set_registry_uri("databricks-uc")

def apply_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
    """
    Applies model to data for a particular station, represented as a pandas DataFrame
    """
    mlflow.set_registry_uri("databricks-uc")

    df_pandas = df_pandas.dropna(axis="index")

    station_id = df_pandas["station_id"].iloc[0]

    model_path = f"models:/{model_name}_{station_id}@Champion"

    input_columns = df_pandas.drop(columns=["station_id", "ts"]).columns
    X = df_pandas[input_columns]

    model = mlflow.sklearn.load_model(model_path)
    prediction = model.predict(X)

    return_df = pd.DataFrame({
        "station_id": df_pandas["station_id"],
        "ts": df_pandas["ts"],
        "prediction": prediction
    })
    return return_df

prediction_df = (features
  .groupby("station_id")
  .applyInPandas(apply_model, schema=apply_return_schema)
  .withColumn("prediction_valid", F.col("ts") + F.make_interval(hours=F.lit(4))) # add four hours for when the label is predicted for
)
display(prediction_df)

# COMMAND ----------

prediction_df.write.saveAsTable(schema + ".predictions")

# COMMAND ----------

# MAGIC %md
# MAGIC Write the newest, most current results into another table that can be queried

# COMMAND ----------

most_current_timestamp = prediction_df.select(F.max("ts")).collect()[0][0]

current_predictions = (prediction_df
  .filter(F.col("ts") == most_current_timestamp)
  .write
  .mode("overwrite")
  .saveAsTable(schema + ".current_predictions")
)

# COMMAND ----------


