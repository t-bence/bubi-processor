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

model_name = schema + ".model"

import mlflow
mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC Load the data

# COMMAND ----------

data = (spark
  .read
  .table(schema + ".features")
  .join(spark.read.table(schema + ".label"), ["station_id", "ts"])
  .na.drop(how="any")
)

data.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the training function
# MAGIC
# MAGIC This will be applied to grouped pandas dataframes that contain the history for one JSON.
# MAGIC It should log a model for that in UC, where the model alias should be the station ID as a string.

# COMMAND ----------

from math import sqrt
import time

import mlflow.sklearn
from mlflow.models import infer_signature


import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

return_schema = "station_id int, r2 double, mse double, rmse double"

def train_model(df: pd.DataFrame) -> pd.DataFrame:
  station_id = df["station_id"].iloc[0]
  
  # start an MLFlow run
  with mlflow.start_run(nested=True, run_name=f"Station {station_id}") as mlflow_run:
    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.set_registry_uri("databricks-uc")
    mlflow.sklearn.autolog(disable=True)

    # set up our model estimator using tuned hyperparameters
    gbr = GradientBoostingRegressor(n_estimators=140, max_depth=34, min_samples_leaf=14)

    X = df[df.drop(columns=["station_id", "ts", "label"]).columns]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    # fit the model on all training data
    gbr_eval_model = gbr.fit(X_train, y_train)

    # evaluate the model on the test set
    y_pred = gbr_eval_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_mse = mean_squared_error(y_test, y_pred)

    # retrain on all data
    gbr_model = gbr.fit(X, y)

    signature = infer_signature(X, y)
    example = X[:3]

    registered_model_name = model_name + "_" + str(station_id)

    # log the model
    mlflow.sklearn.log_model(
      gbr_model,
      artifact_path = str(station_id),
      signature = signature,
      input_example = example,
      registered_model_name = registered_model_name)
    
    mlflow.log_metrics({
      "test_r2": test_r2,
      "test_mse": test_mse,
      "test_rmse": sqrt(test_mse)
    })
    
    # set the station ID as alias to the model
    # time.sleep(10) # Wait 10secs for model version to be created
    # client.set_registered_model_alias(
    #  registered_model_name,
    #  "Champion",
    #  get_latest_model_version(registered_model_name)
    #)

    # return the training metrics as a dataframe
    return pd.DataFrame([[station_id, test_r2, test_mse, sqrt(test_mse)]], 
        columns=["station_id", "r2", "mse", "rmse"])

    

# COMMAND ----------

# MAGIC %md
# MAGIC Perform the training -- this will be lazy, too

# COMMAND ----------

with mlflow.start_run(run_name="Bubi training") as outer_run:
  metrics_df = (data
    .groupBy("station_id")
    .applyInPandas(train_model, schema=return_schema)
    .cache()
  )

# COMMAND ----------

metrics_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # TODO
# MAGIC
# MAGIC Edit train function above so that it assigns Champion label to the current model!
# MAGIC
# MAGIC For now, it will be done separately below...

# COMMAND ----------

station_ids = [row.station_id for row in data.select("station_id").distinct().collect()]

# COMMAND ----------

# helper function that we will use for getting latest version of a model
def get_latest_model_version(model_name):
    """Helper function to get latest model version"""
    model_version_infos = client.search_model_versions("name = '%s'" % model_name)
    return max([model_version_info.version for model_version_info in model_version_infos])
  


for id in station_ids:
  current_name = model_name + "_" + str(id)
  client.set_registered_model_alias(current_name, "Champion", get_latest_model_version(current_name))

# COMMAND ----------


