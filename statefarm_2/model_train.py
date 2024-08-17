from helper_functions import FeatureClean
from helper_functions import FeatureSelect
from helper_functions import pandas_transform
from joblib import dump
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import statsmodels.api as stats_model


DATA_PATH = "./data/"

data_tr = pd.read_csv(DATA_PATH + "./exercise_26_train.csv")
data_te = pd.read_csv(DATA_PATH + "./exercise_26_test.csv")

# fmt: off
# Create the training, validation, and test sets for training the initial L1
# logistic regression model used in feature selection.
x_train, x_val, y_train, y_val = train_test_split(
    data_tr.drop(columns=["y"]),
    data_tr["y"],
    test_size=0.1,
    random_state=13,
)
x_train, x_test, y_train, y_test = train_test_split(
    x_train,
    y_train,
    test_size=4000,
    random_state=13
)

train = pd.concat(
    [x_train, y_train],
    axis=1,
    sort=False
).reset_index(drop=True)

val = pd.concat(
    [x_val, y_val],
    axis=1,
    sort=False
).reset_index(drop=True)

test = pd.concat(
    [x_test, y_test],
    axis=1,
    sort=False
).reset_index(drop=True)

# Remove special characters and clean up variables using helper function.
train = FeatureClean().transform(input_df=train)
val = FeatureClean().transform(input_df=val)
test = FeatureClean().transform(input_df=test)

# Create mean imputation and standard scaler objects for model pipeline.
CAT_COLS = ["x5", "x31", "x81", "x82"]
FLOAT_COLS = train.drop(columns=CAT_COLS + ["y"]).columns.values.tolist()

# Column transformer allows all data to be passed through pipeline.
# This pipeline performs mean imputation, standard scaling, and (OHE) one hot
# encoding, which creates dummy variable columns for categorical features.
impute_scale = Pipeline(steps = [
    ("mean_imputation", SimpleImputer(missing_values=np.nan, strategy="mean")),
    ("std_scaling", StandardScaler())
])

transformation_pipeline = ColumnTransformer(transformers = [
        ("impute_scale", impute_scale, FLOAT_COLS),
        ("one_hot_encode", OneHotEncoder(drop="first", sparse=False), CAT_COLS)
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

transformation_pipeline.fit(train.drop(columns=["y"]))
train_imputed = pandas_transform(
    array_in = transformation_pipeline.transform(train.drop(columns=["y"])),
    pipeline_in = transformation_pipeline
)
train_imputed = pd.concat([train_imputed, train["y"]], axis=1, sort=False)

# Train the intial L1 penalized logistic regression used for feature selection
# by running feature_selection method. Store the selected feature names.
exploratory_LR = LogisticRegression(
    penalty="l1",
    fit_intercept=False,
    solver="liblinear"
)
exploratory_LR.fit(
    train_imputed.drop(columns=["y"]),
    train_imputed["y"]
)

# Feature selection is performed by filtering the top 25 squared coefficients.
exploratory_results = pd.DataFrame(
    train_imputed.drop(columns=["y"]).columns
).rename(columns={0: "name"})
exploratory_results["coefs"] = exploratory_LR.coef_[0]
exploratory_results["coefs_squared"] = exploratory_results["coefs"] ** 2
var_reduced = exploratory_results.nlargest(25, "coefs_squared")
select_features = var_reduced["name"].to_list()

# Train the final model as part of a pipeline and save as a .pkl object. The
# transformation_pipeline is already fitted on the train data. This pipeline
# performs all necessary data transformations and feature selection.
feature_selection = FeatureSelect(
    full_features = exploratory_results["name"].tolist(),
    select_features = select_features
)

feature_pipeline = Pipeline(
    steps=[
        ("feature_cleaning", FeatureClean()),
        ("transformation_pipeline", transformation_pipeline),
        ("feature_selection", feature_selection)
    ]
)

# fmt: on
dump(feature_pipeline, "feature_pipeline.pkl")

# Use all of the training data to fit the logistic model with select features.
all_train = feature_pipeline.transform(data_tr.drop(columns=["y"]))
model = stats_model.Logit(data_tr["y"], all_train).fit()

print(model.summary())

# Save final trained model as a .pkl object.
dump(model, "model.pkl")
