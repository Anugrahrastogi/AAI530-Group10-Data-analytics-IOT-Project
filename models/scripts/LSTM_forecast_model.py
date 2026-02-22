"""
============================================================
LSTM MODEL FOR IOT SENSOR FORECASTING
============================================================

Why LSTM Was Chosen:
--------------------
Long Short-Term Memory (LSTM) is a type of Recurrent Neural Network (RNN)
designed to model sequential and time-series data.

LSTM is particularly effective for:
1. Learning nonlinear temporal relationships
2. Capturing long-term dependencies in sequential data
3. Handling structural changes and regime shifts
4. Modeling complex patterns that classical models may miss

In this IoT environmental dataset:
- Sensors exhibit structural breaks (level shifts)
- Temperature shows nonlinear transitions
- Some sensors display abrupt regime changes
- Time dependencies span multiple days

Unlike Prophet (which models trend + seasonality additively),
LSTM learns patterns directly from historical sequences
without assuming a predefined structure.

What This Script Does:
----------------------
1. Loads IoT sensor dataset
2. Cleans datetime and numeric values
3. Resamples temperature readings to hourly frequency
4. Splits data into:
      - Training set (all except last 7 days)
      - Test set (last 7 days)
5. Scales data using MinMax normalization
6. Converts time-series into supervised learning sequences
7. Builds a deep LSTM neural network
8. Trains model using early stopping
9. Forecasts next 7 days
10. Computes performance metrics (MAE, RMSE)
11. Saves forecast plots per sensor
12. Exports summary CSV of performance metrics

Objective:
----------
Forecast hourly temperature for each sensor (moteid)
over the final 7-day evaluation window.

============================================================
"""

# ------------------------------------------------------------
# Required Libraries
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 1. Load and Prepare Data
# ------------------------------------------------------------

csv_path = "/Users/5212068/Downloads/sensor_data.csv"

# Read dataset
df = pd.read_csv(csv_path)

# Combine date and time into datetime column
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')

# Remove invalid timestamps and sort chronologically
df = df.dropna(subset=['datetime']).sort_values('datetime')

# Convert relevant columns to numeric format
for col in ['temperature','humidity','light','voltage','epoch','moteid']:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# ------------------------------------------------------------
# 2. Create Output Folders
# ------------------------------------------------------------

output_dir = "lstm_results"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

# List to store results for all sensors
results = []


# ------------------------------------------------------------
# 3. Function: Create Sequential Data
# ------------------------------------------------------------
# Converts time-series into supervised learning format:
# Input  -> past N hours
# Output -> next hour

def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)


# ------------------------------------------------------------
# 4. Loop Through All Sensors
# ------------------------------------------------------------

mote_ids = df['moteid'].unique()

for mote in mote_ids:

    # Filter data for current sensor
    mote_df = df[df['moteid'] == mote].copy()
    mote_df = mote_df.set_index('datetime')

    # Resample temperature to hourly mean
    hourly = mote_df['temperature'].resample('h').mean().dropna()

    # Skip sensors with insufficient data
    if len(hourly) < 24*14:
        print(f"Skipping mote {mote} (insufficient data)")
        continue

    # --------------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------------
    # Last 7 days used for evaluation
    test_hours = 24*7
    train = hourly.iloc[:-test_hours]
    test = hourly.iloc[-test_hours:]

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------
    # Normalize data to [0,1] range for stable neural network training
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1,1))
    test_scaled = scaler.transform(test.values.reshape(-1,1))

    # --------------------------------------------------------
    # Create Input Sequences
    # --------------------------------------------------------
    # Use past 72 hours (3 days) to predict next hour
    lookback = 72

    X_train, y_train = create_sequences(train_scaled, lookback)

    # Combine tail of train with test for proper sequence generation
    combined = np.vstack([train_scaled[-lookback:], test_scaled])
    X_test, y_test = create_sequences(combined, lookback)

    if len(X_train) == 0 or len(X_test) == 0:
        continue

    # --------------------------------------------------------
    # Build LSTM Model Architecture
    # --------------------------------------------------------

    model = Sequential([
        Input(shape=(lookback,1)),     # Input shape: (timesteps, features)

        LSTM(128, return_sequences=True),  # First LSTM layer
        Dropout(0.2),                      # Prevent overfitting

        LSTM(64),                          # Second LSTM layer
        Dropout(0.2),

        Dense(32, activation='relu'),      # Fully connected layer
        Dense(1)                           # Output layer (1-step forecast)
    ])

    # Compile model using Adam optimizer and MSE loss
    model.compile(optimizer='adam', loss='mse')

    # Early stopping prevents overfitting
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)

    # --------------------------------------------------------
    # Train Model
    # --------------------------------------------------------
    model.fit(
        X_train, y_train,
        epochs=40,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0
    )

    # --------------------------------------------------------
    # Forecast
    # --------------------------------------------------------
    pred_scaled = model.predict(X_test, verbose=0)

    # Convert scaled predictions back to original scale
    pred = scaler.inverse_transform(pred_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1,1))

    # --------------------------------------------------------
    # Model Evaluation
    # --------------------------------------------------------
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))

    results.append({
        "moteid": mote,
        "mae": mae,
        "rmse": rmse
    })

    # --------------------------------------------------------
    # Save Forecast Plot
    # --------------------------------------------------------
    plt.figure(figsize=(10,4))
    plt.plot(test.index, actual, label="Actual")
    plt.plot(test.index, pred, label="LSTM Forecast")
    plt.title(f"LSTM Forecast - Mote {mote}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", f"mote_{mote}.png"))
    plt.close()

    print(f"Completed mote {mote} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")


# ------------------------------------------------------------
# 5. Save Performance Metrics
# ------------------------------------------------------------

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, "lstm_performance_all_motes.csv"), index=False)

print("\nAll LSTM forecasts completed.")