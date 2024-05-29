# Databricks notebook source
# MAGIC %md
# MAGIC # Construct label

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

hours_in_future = 4

seconds = hours_in_future * 3600

from pyspark.sql.window import Window

windowSpec = (Window
  .partitionBy("station_id")
  .orderBy(F.unix_timestamp(F.col("ts")))
  .rangeBetween(seconds - 10, seconds + 10)
)

label = (features
  .select("station_id", "ts", "bikes")
  .withColumn("label", F.mean("bikes").over(windowSpec))
  .drop("bikes")
)

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
        #feature_name='account_creation_date',
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


from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


# COMMAND ----------

# MAGIC %md
# MAGIC We will do the hyperparameter training on one station only

# COMMAND ----------

selected_station_id = 2100

# COMMAND ----------

# MAGIC %md
# MAGIC Load data and do train-validation-test split. Convert all integer columns to double and simply drop any null values to avoid integer null problems

# COMMAND ----------

from pyspark.sql.types import IntegerType, LongType

all_data = training_set.load_df().filter(F.col("label").isNotNull()).filter(F.col("station_id") == selected_station_id)

integer_columns = [x.name for x in all_data.schema.fields if (x.dataType == IntegerType() or x.dataType == LongType())]

for c in integer_columns:
    all_data = all_data.withColumn(c, F.col(c).cast("double"))

all_data = (all_data
    .na.drop(how="any")
    .drop("station_id")
)

train_df, val_df, test_df = all_data.randomSplit([.6, .2, .2], seed=42)

# COMMAND ----------

all_data.count()

# COMMAND ----------

assembler_inputs = ["bikes", "weekday", "hour", "tenminute", "close_bikes_1", "close_bikes_2", "close_bikes_3", "close_bikes_4", "close_bikes_5"]

vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

rf = RandomForestRegressor(labelCol="label", maxBins=40, seed=42)

pipeline = Pipeline(stages=[vec_assembler, rf])

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="rmse")

# COMMAND ----------

def objective_function(params):    
  # set the hyperparameters that we want to tune
  max_depth = params["max_depth"]
  num_trees = params["num_trees"]

  with mlflow.start_run():
    estimator = pipeline.copy({rf.maxDepth: max_depth, rf.numTrees: num_trees})
    model = estimator.fit(train_df)

    preds = model.transform(val_df)
    rmse = regression_evaluator.setMetricName("rmse").evaluate(preds)
    mlflow.log_metric("rmse", rmse)
    r2 = regression_evaluator.setMetricName("r2").evaluate(preds)
    mlflow.log_metric("r2", r2)

  return rmse

# COMMAND ----------

from hyperopt import hp

search_space = {
  "max_depth": hp.quniform("max_depth", 4, 7, 1),
  "num_trees": hp.quniform("num_trees", 10, 100, 1)
}

# COMMAND ----------

from hyperopt import fmin, tpe, Trials, SparkTrials
import numpy as np
import mlflow
import mlflow.spark
mlflow.pyspark.ml.autolog(log_models=False)

num_evals = 20
trials = Trials()
best_hyperparam = fmin(fn=objective_function, 
                       space=search_space,
                       algo=tpe.suggest, 
                       max_evals=num_evals,
                       trials=trials,
                       rstate=np.random.default_rng(42))



# COMMAND ----------

# MAGIC %md
# MAGIC Retrain model on train & validation dataset and evaluate on test dataset
# MAGIC

# COMMAND ----------

best_hyperparam

# COMMAND ----------

# MAGIC %md
# MAGIC Compute test metrics

# COMMAND ----------

best_max_depth = best_hyperparam["max_depth"]
best_num_trees = best_hyperparam["num_trees"]
best_estimator = pipeline.copy({rf.maxDepth: best_max_depth, rf.numTrees: best_num_trees})

combined_df = train_df.union(val_df) # Combine train & validation together



# COMMAND ----------

with mlflow.start_run():

  model = best_estimator.fit(combined_df)

  pred_df = model.transform(test_df)
  rmse = regression_evaluator.setMetricName("rmse").evaluate(pred_df)
  r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)

  # Log param and metrics for the final model
  mlflow.log_param("maxDepth", best_max_depth)
  mlflow.log_param("numTrees", best_num_trees)
  mlflow.log_metric("test_rmse", rmse)
  mlflow.log_metric("test_r2", r2)
    
  fe.log_model(
    model=model,
    artifact_path="best_model",
    flavor=mlflow.spark,
    training_set=training_set
#    registered_model_name="bence_toth.bubi.bubi_rf_model"
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Train one model for each station, package the model into a custom Pyfunc, then register it

# COMMAND ----------

model_name = "bence_toth.bubi.rf_model"

# COMMAND ----------

# MAGIC %md
# MAGIC The data to train on

# COMMAND ----------

all_stations_data = (training_set
  .load_df()
  .filter(F.col("label").isNotNull())
  .na.drop(how="any")
)

for c in integer_columns:
  all_stations_data = all_stations_data.withColumn(c, F.col(c).cast("double"))

# COMMAND ----------

# MAGIC %md
# MAGIC Train a model for each station

# COMMAND ----------

station_ids = [selected_station_id]

station_to_model = dict()

for station_id in station_ids:

  station_data = all_stations_data.filter(F.col("station_id") == station_id).drop("station_id")

  model = best_estimator.copy().fit(station_data)

  station_to_model[str(station_id)] = model


# COMMAND ----------

# MAGIC %md
# MAGIC Create the custom model

# COMMAND ----------

class StationDelegatingModel(mlflow.pyfunc.PythonModel):
  def __init__(self, station_to_model_map):
    self.station_to_model_map = station_to_model_map

  def predict(self, context, model_input, params=None):
    # get the model -- if it does not exists, just throw the error
    model = self.station_to_model_map[str(model_input)]

    return model.predict()

# COMMAND ----------

# MAGIC %md
# MAGIC Log the custom model

# COMMAND ----------

from mlflow.models import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec

# Option 1: Manually construct the signature object
input_schema = Schema(
    [
        ColSpec("string", "station id")
    ]
)

with mlflow.start_run() as run:

  mlflow.pyfunc.log_model(
    python_model=StationDelegatingModel(station_to_model),
    artifact_path="rf_model",
    signature=ModelSignature(inputs=input_schema, outputs=input_schema),
    registered_model_name=model_name
  )

# COMMAND ----------


