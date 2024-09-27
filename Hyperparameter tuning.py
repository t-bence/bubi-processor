# Databricks notebook source
# MAGIC %md
# MAGIC # Hyperparameter tuning
# MAGIC
# MAGIC This notebook contains an error: there should be no CV for hyperparameter tuning as it can lead to data leakage!

# COMMAND ----------

# MAGIC %sql 
# MAGIC USE CATALOG bence_toth;
# MAGIC USE SCHEMA bubi_project;

# COMMAND ----------

import pyspark.sql.functions as F

schema = "bence_toth.bubi_project"

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

features = fe.read_table(name=schema + ".features")

# COMMAND ----------

# MAGIC %md
# MAGIC Label: number of bikes at the same station, four hours later

# COMMAND ----------

label = spark.read.table(schema + ".label")

# COMMAND ----------

label.printSchema()

# COMMAND ----------

label.orderBy("station_id", F.col("ts").desc()).display()

# COMMAND ----------

# MAGIC %md
# MAGIC Create training set from feature store

# COMMAND ----------

from databricks.feature_engineering.entities import FeatureLookup

# COMMAND ----------

feature_lookups = [
    FeatureLookup(
        table_name="bence_toth.bubi_project.features",
        lookup_key="station_id",
        timestamp_lookup_key="ts"
    )
]

training_set = fe.create_training_set(
        df=label,
        feature_lookups=feature_lookups,
        label="label",
        exclude_columns="ts"
    )

# COMMAND ----------

training_set.load_df().display()

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")


# COMMAND ----------

# MAGIC %md
# MAGIC We will do the hyperparameter training on one station only

# COMMAND ----------

selected_station_id = 2100

# COMMAND ----------

# MAGIC %md
# MAGIC Convert all integer columns to double and simply drop any null values to avoid integer null problems

# COMMAND ----------

from pyspark.sql.types import IntegerType, LongType

all_data = training_set.load_df().filter(F.col("label").isNotNull()).filter(F.col("station_id") == selected_station_id)

all_data = (all_data
    .na.drop(how="any")
    .drop("station_id")
    .toPandas()
)

# COMMAND ----------

# MAGIC %md
# MAGIC Train - test split

# COMMAND ----------

from sklearn.model_selection import train_test_split

print(f"We have {all_data.shape[0]} records in our source dataset")

# split target variable into it's own dataset
target_col = "label"
X_all = all_data.drop(labels=target_col, axis=1)
y_all = all_data[target_col]

# test / train split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=0.90, random_state=42)
print(f"We have {X_train.shape[0]} records in our training dataset")
print(f"We have {X_test.shape[0]} records in our test dataset")

# COMMAND ----------

from hyperopt import hp

param_space = {
  'max_depth': hp.uniformint('dtree_max_depth_int', 5, 50),
  'n_estimators': hp.uniformint('dtree_min_samples_split', 10, 150),
  'min_samples_leaf': hp.uniformint('dtree_min_samples_leaf', 1, 20)
}

# COMMAND ----------

# MAGIC %md
# MAGIC Define the optimization function

# COMMAND ----------

from math import sqrt

import mlflow
import mlflow.data
import mlflow.sklearn

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_validate

from hyperopt import STATUS_OK

def tuning_objective(params):
  # start an MLFlow run
  with mlflow.start_run(nested=True) as mlflow_run:
    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        disable=False,
        log_input_examples=True,
        silent=True,
        exclusive=False)

    # set up our model estimator
    gbr = GradientBoostingRegressor(**params)
    
    # cross-validated on the training set
    validation_scores = ["r2", "neg_root_mean_squared_error"]
    cv_results = cross_validate(gbr, 
                                X_train, 
                                y_train, 
                                cv=5,
                                scoring=validation_scores)
    # log the average cross-validated results
    cv_score_results = {}
    for val_score in validation_scores:
      cv_score_results[val_score] = cv_results[f'test_{val_score}'].mean()
      mlflow.log_metric(f"cv_{val_score}", cv_score_results[val_score])

    # fit the model on all training data
    gbr_model = gbr.fit(X_train, y_train)

    # evaluate the model on the test set
    y_pred = gbr_model.predict(X_test)
    r2_score(y_test, y_pred)
    mean_squared_error(y_test, y_pred)
    
    # return the negative of our cross-validated F1 score as the loss
    return {
      "loss": -cv_score_results["neg_root_mean_squared_error"],
      "status": STATUS_OK,
      "run": mlflow_run
    }

# COMMAND ----------

from hyperopt import SparkTrials, fmin, tpe

# set the path for mlflow experiment
mlflow.set_experiment(f"/Workspace/Users/bence.toth@datapao.com/Bubi-tuning")

trials = SparkTrials(parallelism=4)
with mlflow.start_run(run_name="Model Tuning with Hyperopt Demo") as parent_run:
  fmin(tuning_objective,
      space=param_space,
      algo=tpe.suggest,
      max_evals=20,  # Increase this when widening the hyperparameter search space.
      trials=trials)

best_result = trials.best_trial["result"]
best_run = best_result["run"]

# COMMAND ----------

# MAGIC %md
# MAGIC Best results were achieved with:
# MAGIC [see experiment](https://adb-3679152566148441.1.azuredatabricks.net/ml/experiments/3561046141604344?o=3679152566148441&searchFilter=&orderByKey=metrics.%60cv_neg_root_mean_squared_error%60&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D)
# MAGIC
# MAGIC - n_estimators = 140
# MAGIC - max_depth = 34
# MAGIC - min_samples_leaf = 14

# COMMAND ----------

# MAGIC %md
# MAGIC Create the custom model

# COMMAND ----------

"""
class StationDelegatingModel(mlflow.pyfunc.PythonModel):
  def __init__(self, station_to_model_map):
    self.station_to_model_map = station_to_model_map

  def predict(self, context, model_input, params=None):
    # get the model -- if it does not exists, just throw the error
    model = self.station_to_model_map[str(model_input)]

    return model.predict()
"""
