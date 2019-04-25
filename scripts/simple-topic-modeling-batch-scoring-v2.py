#!/usr/bin/python

import sys, os, pickle
import pandas as pd
from sklearn import model_selection
from sklearn.externals import joblib
import dsx_core_utils, re, jaydebeapi
from sqlalchemy import *
from sqlalchemy.types import String, Boolean

import json

# setup dsxr environmental vars from command line input
from dsx_ml.ml import dsxr_setup_environment
dsxr_setup_environment()

# define variables
args = {'target': '/datasets/train-topic-probabilities.csv', 'remoteHost': '', 'sysparm': '', 'livyVersion': 'livyspark2', 'output_datasource_type': '', 'source': '/datasets/train.csv', 'remoteHostImage': '', 'execution_type': 'DSX', 'output_type': 'Localfile'}
input_data = os.getenv("DEF_DSX_DATASOURCE_INPUT_FILE", (os.getenv("DSX_PROJECT_DIR") + args.get("source")))
output_data = os.getenv("DEF_DSX_DATASOURCE_OUTPUT_FILE", (os.getenv("DSX_PROJECT_DIR") + args.get("target")))
# model_path = os.getenv("DSX_PROJECT_DIR") + os.path.join("/models", os.getenv("DSX_MODEL_NAME","simple-topic-modeling"), os.getenv("DSX_MODEL_VERSION","2"),"model")

# this section was modified to match online scoring scripts style:
model_name = "simple-topic-modeling"
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
dataframe = pd.read_csv(input_data)

# load the model from disk
loaded_model = joblib.load(open(model_path, 'rb'))

# predictions
scoring_result = loaded_model.transform(dataframe)

# save scoring result to given target
scoring_df = pd.DataFrame(scoring_result)
scoring_df.columns = ["topic_" + str(i) for i in range(len(scoring_df.columns))]

# save output to csv
scoring_df.to_csv(output_data, encoding='utf-8', index = False)