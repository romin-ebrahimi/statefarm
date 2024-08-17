# fmt: off
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from joblib import load
import numpy as np
import pandas as pd
from typing import Union

# Create the Python Fast API.
api = FastAPI()

# Load model and feature preprocessing pipeline into memory.
model = load("model.pkl")
feature_pipeline = load("feature_pipeline.pkl")

# Post request for the classification model is at http://localhost:1313/predict
@api.post("/predict")
async def predict(data: Union[dict, list[dict]]) -> list[dict]:
    """
    Given a JSON with features {x0, x1, ..., x99}, the model pipeline creates
    the features and generates probability estimates. Then, the returned JSON
    contains estimates mapped to 'business_outcome' in {0, 1}, 'phat' (the raw
    probability estimates), and the 25 variables that the model uses sorted in
    alphabetical order.
    Args:
        data: containing the input features as {x0, x1, ..., x99} or in the case
        of multiple [{x0, x1, ..., x99}, {x0, x1, ..., x99}].
    Returns:
        JSON containing business_outcome, phat, and 25 features in alphabetical
        order. e.g. {"business_outcome": 0, "phat": 0.40, "x0": 0, "x1": 0, ...,
        "x99": 0}. Returned payload can be for one observation or N observations
        in an array format.
    """
    # Hardcoded classification threshold is the 75th percentile of 'phat'.
    CLASS_THRESHOLD = 0.712

    # Feature pipeline calls preprocessing functions that will clean the data,
    # apply mean imputation, apply standardization, create one hot encoded
    # dummy variables for categorical features, and select the 25 features that
    # the trained model needs.
    if isinstance(data, dict):
        features = feature_pipeline.transform(pd.DataFrame([data]))
    else:
        features = feature_pipeline.transform(pd.DataFrame.from_records(data))

    # Generate 'phat' probability estimates.
    phat = model.predict(features)
    features["phat"] = phat

    # Generate business outcome estimates {0, 1}. Threshold is hardcoded.
    business_outcome = np.zeros(phat.shape[0])
    business_outcome[phat >= CLASS_THRESHOLD] = 1
    features["business_outcome"] = business_outcome.astype(int)

    # For the returned body, features are reordered in alphabetical order.
    features = features.reindex(sorted(features.columns), axis=1)

    return JSONResponse(features.to_dict(orient="records"))
