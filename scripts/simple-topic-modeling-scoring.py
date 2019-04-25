# Copyright 2017, 2018 IBM. IPLA licensed Sample Materials.


import pandas as pd
from sklearn import model_selection
from sklearn.externals import joblib
import pickle
import sys
import os, json
from collections import OrderedDict

import os
global model
model = None


def init():
    global model
    global serialization_method

    model_name = "simple-topic-modeling"
    version = "latest"
    project_name = os.environ.get("DSX_PROJECT_NAME")
    user_id = os.environ.get("DSX_USER_ID", "990")
    project_path = "/user-home/" + user_id + "/DSX_Projects/" + project_name
    model_parent_path = project_path + "/models/" + model_name + "/"
    metadata_path = model_parent_path + "metadata.json"

    # fetch info from metadata.json
    with open(metadata_path) as data_file:
        meta_data = json.load(data_file)

    # if latest version, find latest version from  metadata.json
    if (version == "latest"):
        version = meta_data.get("latestModelVersion")

    # prepare model path using model name and version
    model_path = model_parent_path + str(version) + "/model"

    serialization_method = "joblib"

    # load model
    if serialization_method == "joblib":
        model = joblib.load(open(model_path, 'rb'))
    elif serialization_method == "pickle":
        model = pickle.load(open(model_path, 'rb'))


def score(args):
    global model

    input_json = args.get("input_json")

    # convert to pandas dataframe
    data_frame = pd.DataFrame.from_dict(input_json)

    # scoring method
    classes = None
    probabilities = None
    predictions = model.transform(data_frame).tolist()

    try:
        if hasattr(model, 'classes_'):
            classes = model.classes_.tolist()
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(data_frame).tolist()
    except:
        pass

    return {
        "classes": classes,
        "probabilities": probabilities,
        "predictions": predictions
    }

def test_score(args):
    """Call this method to score in development."""
    init()
    return score(args)