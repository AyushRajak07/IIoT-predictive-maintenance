"""
predict_live.py

Purpose: Simulate ONE new live sensor reading and send it to your deployed
Azure ML endpoint (see README Part G) to get a real-time failure prediction.

Before running: fill in ENDPOINT_URL and API_KEY below with the values
from Azure ML Studio -> Endpoints -> your endpoint -> Consume tab.
"""

import requests   # lets Python make HTTP requests (talk to web APIs) -- pip install requests
import json       # formats our data as JSON, the text format APIs expect
import numpy as np  # for generating a realistic random reading

# ---------------------------------------------------------------------------
# STEP 1: Fill in your endpoint details (from Azure ML Studio -> Consume tab)
# ---------------------------------------------------------------------------
ENDPOINT_URL = "https://iiot-failure-endpoint.<region>.inference.ml.azure.com/score"  # replace with your real URL
API_KEY = "<your-api-key-here>"  # replace with your real key -- keep this secret, don't commit it to git

# ---------------------------------------------------------------------------
# STEP 2: Generate one new simulated sensor reading
# ---------------------------------------------------------------------------
# Here we hand-craft a reading that LOOKS like early warning signs
# (elevated vibration/temperature/pressure) to test that the model
# correctly flags risk. Change these numbers to test different scenarios.
new_reading = {
    "vibration_mm_s": 4.8,   # elevated vs. healthy baseline of ~2.0
    "temperature_c": 78.2,   # elevated vs. healthy baseline of ~55.0
    "pressure_bar": 5.9      # slightly elevated vs. healthy baseline of ~5.0
}

# ---------------------------------------------------------------------------
# STEP 3: Format the request the way Azure ML endpoints expect it
# ---------------------------------------------------------------------------
# Azure ML real-time endpoints expect a JSON body with a "data" key containing
# rows of feature values, in the SAME column order used during training.
payload = {
    "data": [
        [new_reading["vibration_mm_s"], new_reading["temperature_c"], new_reading["pressure_bar"]]
    ]
}

# ---------------------------------------------------------------------------
# STEP 4: Send the request
# ---------------------------------------------------------------------------
headers = {
    "Content-Type": "application/json",       # tells the server we're sending JSON
    "Authorization": f"Bearer {API_KEY}"      # proves we're allowed to use this endpoint
}

response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(payload))

# ---------------------------------------------------------------------------
# STEP 5: Interpret the result
# ---------------------------------------------------------------------------
if response.status_code == 200:
    result = response.json()  # parses the JSON response back into a Python object
    prediction = result[0]    # our model returns one prediction per input row; we sent 1 row
    status = "FAILURE RISK" if prediction == 1 else "HEALTHY"
    print(f"Reading: vibration={new_reading['vibration_mm_s']}mm/s, "
          f"temp={new_reading['temperature_c']}C, pressure={new_reading['pressure_bar']}bar")
    print(f"Prediction: {status}")
else:
    # If something goes wrong (bad key, endpoint not ready, wrong format),
    # print the error so you can debug it.
    print(f"Request failed with status {response.status_code}")
    print(response.text)
