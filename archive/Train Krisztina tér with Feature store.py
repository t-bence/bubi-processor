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

gold_table.cache().count()

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

gold_table.count()

# COMMAND ----------

display(gold_table
  .agg(
    F.min("index"),
    F.min("index").cast("int"),
    F.max("index"),
    F.max("index").cast("int")
  )
)

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

fs.write_table(name=krisztina_table_name, df=krisztina_table.drop("future_bikes"), mode="overwrite")

# COMMAND ----------

# MAGIC %md
# MAGIC Get data from Feature store and train model

# COMMAND ----------

label_df = krisztina_table.select("index", "future_bikes")

# COMMAND ----------

from databricks.feature_store import FeatureLookup
from sklearn.model_selection import train_test_split

index = "index"
label = "future_bikes"

model_feature_lookups = [FeatureLookup(table_name=krisztina_table_name, lookup_key=index)]

# fs.create_training_set will look up features in model_feature_lookups with matched key from inference_data_df
training_set = fs.create_training_set(label_df, model_feature_lookups, label=label, exclude_columns=index)
training_pd = training_set.load_df().toPandas()

# display(training_set.load_df())

# Create train and test datasets
X = training_pd.drop(label, axis=1)
y = training_pd[label]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create a validation set from the train set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC # Numerical pipeline

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

num_imputers = []
num_imputers.append(("impute_mean", SimpleImputer(), ["bikes", "close_bikes_1", "close_bikes_2", "close_bikes_3", "close_bikes_4", "close_bikes_5", "hour", "minute", "weekday"]))

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors='coerce'))),
    ("imputers", ColumnTransformer(num_imputers)),
    ("standardizer", StandardScaler()),
])

numerical_transformers = [("numerical", numerical_pipeline, ["close_bikes_5", "close_bikes_3", "close_bikes_1", "hour", "bikes", "weekday", "close_bikes_4", "close_bikes_2", "minute"])]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Categorical pipeline

# COMMAND ----------

from databricks.automl_runtime.sklearn import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

one_hot_imputers = []

one_hot_pipeline = Pipeline(steps=[
    ("imputers", ColumnTransformer(one_hot_imputers, remainder="passthrough")),
    ("one_hot_encoder", OneHotEncoder(handle_unknown="indicator")),
])

categorical_one_hot_transformers = [("onehot", one_hot_pipeline, ["minute", "weekday"])]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merge pipelines

# COMMAND ----------

from sklearn.compose import ColumnTransformer

transformers = numerical_transformers + categorical_one_hot_transformers

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's to do a parameter study for this sample and use these params for all stations later on.
# MAGIC
# MAGIC Code is based on an AutoML notebook, where XGBoost performed best.

# COMMAND ----------

from xgboost import XGBRegressor

# COMMAND ----------

import mlflow
from mlflow.models import Model, infer_signature, ModelSignature
from mlflow.pyfunc import PyFuncModel
from mlflow import pyfunc
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials


# Create a separate pipeline to transform the validation dataset. This is used for early stopping.
pipeline_val = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
])

mlflow.sklearn.autolog(disable=True)
pipeline_val.fit(X_train, y_train)
X_val_processed = pipeline_val.transform(X_val)

def objective(params):
  with mlflow.start_run(experiment_id="XGB_krisztina_hyperopt") as mlflow_run:
    xgb_regressor = XGBRegressor(**params)

    model = Pipeline([
        ("column_selector", col_selector),
        ("preprocessor", preprocessor),
        ("regressor", xgb_regressor),
    ])

    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        log_input_examples=True,
        silent=True,
    )

    model.fit(X_train, y_train, regressor__early_stopping_rounds=5, regressor__verbose=False, regressor__eval_set=[(X_val_processed,y_val)])

    
    # Log metrics for the training set
    mlflow_model = Model()
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
    pyfunc_model = PyFuncModel(model_meta=mlflow_model, model_impl=model)
    training_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_train.assign(**{str(target_col):y_train}),
        targets=target_col,
        model_type="regressor",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "training_"}
    )
    # Log metrics for the validation set
    val_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_val.assign(**{str(target_col):y_val}),
        targets=target_col,
        model_type="regressor",
        evaluator_config= {"log_model_explainability": False,
                           "metric_prefix": "val_"}
   )
    xgb_val_metrics = val_eval_result.metrics
    # Log metrics for the test set
    test_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_test.assign(**{str(target_col):y_test}),
        targets=target_col,
        model_type="regressor",
        evaluator_config= {"log_model_explainability": False,
                           "metric_prefix": "test_"}
   )
    xgb_test_metrics = test_eval_result.metrics

    loss = -xgb_val_metrics["val_r2_score"]

    # Truncate metric key names so they can be displayed together
    xgb_val_metrics = {k.replace("val_", ""): v for k, v in xgb_val_metrics.items()}
    xgb_test_metrics = {k.replace("test_", ""): v for k, v in xgb_test_metrics.items()}

    return {
      "loss": loss,
      "status": STATUS_OK,
      "val_metrics": xgb_val_metrics,
      "test_metrics": xgb_test_metrics,
      "model": model,
      "run": mlflow_run,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC Add the parameter space
# MAGIC Let's not train for everything, because there is no time...

# COMMAND ----------

space = {
  "colsample_bytree": hp.uniform("colsample_bytree", 0.3, 0.9),
  "learning_rate": hp.loguniform("learning_rate", -5, -1),
  "max_depth": hp.quniform("max_depth", 3, 10, 1),
  "min_child_weight": hp.quniform("min_child_weight", 5, 50, 5),
  "n_estimators": hp.quniform("n_estimators", 100, 200, 10),
  "n_jobs": 100,
  "subsample": 0.45416365621556487,
  "verbosity": 0,
  "random_state": 320723541,
}

# COMMAND ----------

from hyperopt import SparkTrials

trials = SparkTrials()
fmin(objective,
     space=space,
     algo=tpe.suggest,
     max_evals=4,  # Increase this when widening the hyperparameter search space.
     trials=trials)

best_result = trials.best_trial["result"]
model = best_result["model"]
mlflow_run = best_result["run"]

display(
  pd.DataFrame(
    [best_result["val_metrics"], best_result["test_metrics"]],
    index=["validation", "test"]))

set_config(display="diagram")
model
