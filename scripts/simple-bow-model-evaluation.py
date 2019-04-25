#!/usr/bin/python

import pandas as pd
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.metrics import *
from keras.utils import to_categorical
import pickle
import json
import time
import sys
import os
from uuid import uuid4
import numpy as np
import dsx_core_utils, requests, re, jaydebeapi
from dsx_ml.ml import save_evaluation_metrics
from pyspark.sql import SparkSession


# setup dsxr environmental vars from command line input
from dsx_ml.ml import dsxr_setup_environment
dsxr_setup_environment()

# define variables
args = {"threshold": {"min_value": 0.3, "metric": "accuracyScore", "mid_value": 0.7}, "dataset": "/datasets/test.csv", "livyVersion": "livyspark2", "execution_type": "DSX", "remoteHostImage": "", "published": "false", "remoteHost": ""}
# model_path = os.path.join(os.getenv("DSX_PROJECT_DIR"), "models", os.getenv("DEF_DSX_MODEL_NAME", "simple-bow"), os.getenv("DEF_DSX_MODEL_VERSION", "2"), "model")

# this section was modified to match online scoring scripts style:
model_name = "simple-bow"
version = "latest"
model_parent_path = os.getenv("DSX_PROJECT_DIR") + os.path.join("/models", os.getenv("DSX_MODEL_NAME", model_name))
metadata_path = os.path.join(model_parent_path, "metadata.json")

# fetch info from metadata.json
with open(metadata_path) as data_file:
  meta_data = json.load(data_file)

# if latest version, find latest version from  metadata.json
if (version == "latest"):
  version = meta_data.get("latestModelVersion")

# prepare model path using model name and version
model_path = os.path.join(model_parent_path, str(version), "model")

# the rest of the script is similar to the regular batch scoring template:

# load the input data

input_data = os.getenv("DEF_DSX_DATASOURCE_INPUT_FILE", os.getenv("DSX_PROJECT_DIR") + args.get("dataset"))
df1 = pd.read_csv(input_data)

X = df1[['review']]
y_true = df1[['pos']].values.flatten()

# load the model from disk 
loaded_model = joblib.load(open(model_path, 'rb'))

# predictions
startTime = int(time.time())
y_pred = loaded_model.predict(X)

# classification Metrics
threshold = {'min_value': 0.3, 'metric': 'accuracyScore', 'mid_value': 0.7}

if (len(y_true.shape) == 1):
    y_true_1d = y_true
    y_true_2d = to_categorical(y_true)

else:
    y_true_1d = np.argmax(y_true, axis=1)
    y_true_2d = y_true

if (len(y_pred.shape) == 1):
    y_pred_1d = y_pred
    y_pred_2d = to_categorical(y_pred)

else:
    y_pred_1d = np.argmax(y_pred, axis=1)
    y_pred_2d = y_pred

eval_fields = {
        "accuracyScore": accuracy_score(y_true_1d, y_pred_1d),
        "precisionScore": precision_score(y_true_1d, y_pred_1d, average="weighted"),
        "recallScore": recall_score(y_true_1d, y_pred_1d, average="weighted"),
        "areaUnderROC": roc_auc_score(y_true_2d, y_pred_2d),
        "thresholdMetric": threshold["metric"],
        "thresholdMinValue": threshold["min_value"],
        "thresholdMidValue": threshold["mid_value"]
    }

# feel free to customize to your own performance logic using the values of "good", "poor", and "fair".
if(eval_fields[eval_fields["thresholdMetric"]] >= threshold.get('mid_value', 0.70)):
    eval_fields["performance"] = "good"
elif(eval_fields[eval_fields["thresholdMetric"]] <= threshold.get('min_value', 0.25)):
    eval_fields["performance"] = "poor"
else:
    eval_fields["performance"] = "fair"

save_evaluation_metrics(eval_fields, "simple-bow", latestModelVersion, startTime)