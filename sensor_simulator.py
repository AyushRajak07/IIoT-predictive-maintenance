"""
sensor_simulator.py

Purpose: Pretend to be a factory machine with three sensors (vibration,
temperature, pressure) and generate a realistic-looking dataset that
includes both "healthy" readings and readings that lead up to a "failure".

We need labeled data (failure = 0 or 1) because Azure ML needs examples
of both outcomes to learn the difference between them. In a real factory
this label would come from maintenance logs; here we simulate it.
"""

import numpy as np          # numpy: generates random numbers efficiently (used for sensor noise)
import pandas as pd         # pandas: builds and saves our data as a table (DataFrame) -> CSV
from datetime import datetime, timedelta  # used to create realistic timestamps

# A fixed random seed means you'll get the SAME "random" data every time you run
# this script. Useful while learning, so your results are reproducible.
np.random.seed(42)

def simulate_machine(machine_id: str, num_readings: int, start_time: datetime):
    """
    Simulates one machine's sensor history.

    Logic: most of the time the machine is healthy, with sensor values
    fluctuating around a normal baseline (this is called 'noise').
    Occasionally we simulate a 'degradation window' -- a stretch of readings
    where vibration/temperature/pressure drift upward, ending in a failure.
    This mimics how real machines behave: they don't fail instantly,
    they show warning signs first.
    """
    rows = []  # will hold one dictionary per sensor reading

    # Baseline "healthy" values -- tweak these to represent a different machine type.
    baseline_vibration = 2.0     # mm/s, typical healthy vibration
    baseline_temperature = 55.0  # Celsius, typical healthy operating temperature
    baseline_pressure = 5.0      # bar, typical healthy pressure

    i = 0
    while i < num_readings:
        # 3% chance to start a "degradation window" leading to failure,
        # as long as we have at least 40 readings left to simulate the ramp-up.
        if np.random.rand() < 0.03 and i < num_readings - 40:
            window_length = np.random.randint(20, 40)  # how many readings until failure
            for step in range(window_length):
                # progress goes from 0.0 (start of degradation) to 1.0 (failure point)
                progress = step / window_length

                # As progress increases, sensor values drift further from baseline.
                # np.random.normal(mean, std) adds realistic random noise on top of the trend.
                vibration = baseline_vibration + progress * 6 + np.random.normal(0, 0.3)
                temperature = baseline_temperature + progress * 30 + np.random.normal(0, 1.5)
                pressure = baseline_pressure + progress * 3 + np.random.normal(0, 0.4)

                # Only the last few readings in the window are labeled as actual
                # "failure" (1) -- the earlier ones are "at risk but not yet failed" (0).
                # This teaches the model to recognize the pattern, not just the final spike.
                label = 1 if progress > 0.85 else 0

                rows.append({
                    "timestamp": start_time + timedelta(minutes=i),
                    "machine_id": machine_id,
                    "vibration_mm_s": round(vibration, 2),
                    "temperature_c": round(temperature, 2),
                    "pressure_bar": round(pressure, 2),
                    "failure": label
                })
                i += 1
                if i >= num_readings:
                    break
        else:
            # Normal healthy reading: baseline value + small random noise.
            vibration = baseline_vibration + np.random.normal(0, 0.3)
            temperature = baseline_temperature + np.random.normal(0, 1.5)
            pressure = baseline_pressure + np.random.normal(0, 0.4)

            rows.append({
                "timestamp": start_time + timedelta(minutes=i),
                "machine_id": machine_id,
                "vibration_mm_s": round(vibration, 2),
                "temperature_c": round(temperature, 2),
                "pressure_bar": round(pressure, 2),
                "failure": 0  # healthy reading, no failure
            })
            i += 1

    return rows

# Build data for a few different simulated machines, so the model learns
# general patterns rather than memorizing one machine's exact numbers.
all_rows = []
machine_ids = ["MACHINE_01", "MACHINE_02", "MACHINE_03"]
start = datetime(2026, 1, 1)  # arbitrary start date for the fake timestamps

for m_id in machine_ids:
    # 2000 readings per machine (~33 hours of "one reading per minute" data)
    all_rows.extend(simulate_machine(m_id, num_readings=2000, start_time=start))

# Convert the list of dictionaries into a pandas DataFrame (a table).
df = pd.DataFrame(all_rows)

# Save the table to a CSV file -- this is the dataset we'll upload to Azure ML.
# index=False means "don't write pandas' internal row numbers into the file."
df.to_csv("machine_sensor_data.csv", index=False)

print(f"Generated {len(df)} rows across {len(machine_ids)} machines.")
print(f"Failure-labeled rows: {df['failure'].sum()} ({df['failure'].mean()*100:.1f}%)")
print("Saved to machine_sensor_data.csv")
