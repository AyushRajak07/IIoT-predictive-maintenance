"""
train_model.py

Purpose: Train a machine learning model that predicts machine failure
(0 = healthy, 1 = failure) from vibration/temperature/pressure readings.

Run this INSIDE an Azure ML Notebook (after uploading this file, or pasting
its cells into a new notebook) so it can access your registered data asset
and save the trained model into your workspace.

If you just want to test the logic locally first, you can also run this
script directly against the local machine_sensor_data.csv produced by
sensor_simulator.py -- see the "LOCAL TEST MODE" section at the bottom.
"""

import pandas as pd                                   # for loading/manipulating the tabular data
from sklearn.model_selection import train_test_split   # splits data into train/test sets
from sklearn.ensemble import RandomForestClassifier     # the ML algorithm we'll use
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import joblib                                          # saves the trained model to a file

# ---------------------------------------------------------------------------
# STEP 1: Load the data
# ---------------------------------------------------------------------------
# In Azure ML Studio, you'd normally load your registered Data Asset like this:
#
#   from azureml.core import Workspace, Dataset
#   ws = Workspace.from_config()                       # connects to your workspace
#   dataset = Dataset.get_by_name(ws, "machine-sensor-data")  # fetches the asset by name
#   df = dataset.to_pandas_dataframe()                  # converts it into a pandas table
#
# For simplicity here (and so you can test locally first), we load directly
# from the CSV file produced by sensor_simulator.py:
df = pd.read_csv("machine_sensor_data.csv")

# ---------------------------------------------------------------------------
# STEP 2: Choose our features (inputs) and target (output to predict)
# ---------------------------------------------------------------------------
# "Features" (X) are the columns the model is allowed to look at to make its
# prediction. "Target" (y) is the column we want it to predict.
feature_columns = ["vibration_mm_s", "temperature_c", "pressure_bar"]
X = df[feature_columns]     # a table with just the 3 sensor columns
y = df["failure"]           # a single column: 0 or 1

# ---------------------------------------------------------------------------
# STEP 3: Split into training data and test data
# ---------------------------------------------------------------------------
# We train the model on 80% of the rows, and hold back 20% that the model
# NEVER sees during training. We use that held-back 20% afterward to check
# how well the model generalizes to new, unseen readings -- this is the
# single most important habit in machine learning: never grade a model
# on the same data it studied from.
#
# stratify=y keeps the same proportion of failure/healthy rows in both
# the train and test sets (important since failures are rare in our data).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# STEP 4: Train the model
# ---------------------------------------------------------------------------
# RandomForestClassifier builds many decision trees (n_estimators=100 means
# 100 trees), each trained on a slightly different random subset of the
# data, then averages their votes. This "wisdom of crowds" approach is
# accurate and resistant to overfitting, and works well on tabular sensor
# data like ours without much tuning -- a great default choice.
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
# class_weight="balanced" tells the model to pay extra attention to the
# rarer class (failures) instead of just optimizing for overall accuracy,
# since failures are only ~6% of our data and we care a lot about catching them.

model.fit(X_train, y_train)  # this is the actual "learning" step

# ---------------------------------------------------------------------------
# STEP 5: Evaluate on the test set (data the model has never seen)
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)  # ask the model to predict on the held-out rows

accuracy = accuracy_score(y_test, y_pred)     # % of predictions that were correct overall
precision = precision_score(y_test, y_pred)   # of the times the model said "failure", how often was it right?
recall = recall_score(y_test, y_pred)         # of all the ACTUAL failures, how many did the model catch?

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print("\nFull report:\n", classification_report(y_test, y_pred))

# Note: for predictive maintenance, RECALL usually matters more than
# precision -- missing a real failure (a false negative) is far more
# costly than a false alarm (a false positive). If recall is low, consider
# lowering the prediction threshold or gathering more failure examples.

# ---------------------------------------------------------------------------
# STEP 6: Save the trained model to a file
# ---------------------------------------------------------------------------
joblib.dump(model, "model.pkl")  # serializes the trained model to disk
print("\nModel saved to model.pkl")

# ---------------------------------------------------------------------------
# STEP 7: Register the model in your Azure ML workspace (run this part
# inside Azure ML Studio's notebook, not on your local machine)
# ---------------------------------------------------------------------------
# Registering makes the model visible in the "Models" tab of Azure ML Studio,
# versioned, and ready to deploy as an endpoint (see README Part G).
#
#   from azureml.core import Workspace, Model
#   ws = Workspace.from_config()
#   registered_model = Model.register(
#       workspace=ws,
#       model_path="model.pkl",              # local path to the file we just saved
#       model_name="iiot-failure-model",      # name it will appear under in Azure ML
#       description="RandomForest predicting machine failure from vibration/temp/pressure"
#   )
#   print(f"Registered model version: {registered_model.version}")
