"""
============================================================
PROPHET MODEL FOR IOT SENSOR FORECASTING
============================================================

Why Prophet Was Chosen:
-----------------------
Prophet is a decomposable time-series forecasting model developed by Facebook.
It is particularly well-suited for time-series data that exhibits:

1. Strong seasonality (daily / weekly patterns)
2. Trend changes or structural breaks
3. Missing values
4. Moderate dataset sizes

In this IoT environmental monitoring dataset:
- Temperature readings show clear daily cycles.
- Several sensors exhibit structural shifts (level jumps).
- Data length per sensor is moderate (~1 month).
- Forecast horizon is short (7 days).

Prophet is appropriate because it models time-series as:

    y(t) = Trend(t) + Seasonality(t) + Noise

It automatically detects:
- Changepoints (trend breaks)
- Daily and weekly seasonal patterns
- Additive or multiplicative seasonal behavior

What This Script Does:
----------------------
1. Loads IoT sensor dataset
2. Cleans and preprocesses datetime + numeric values
3. Resamples sensor readings to hourly frequency
4. Removes extreme outliers (optional spike removal)
5. Splits data into:
      - Training set (all except last 7 days)
      - Test set (last 7 days)
6. Runs hyperparameter grid search for Prophet
7. Selects best configuration based on RMSE
8. Generates forecast plots per sensor
9. Saves performance metrics (MAE, RMSE)
10. Outputs summary CSV file

Objective:
----------
Forecast temperature for each sensor (moteid) over the next 7 days
and evaluate model performance.

============================================================
"""
# ------------------------------------------------------------
# Required Libraries
# ------------------------------------------------------------
# pandas -> data manipulation
# numpy -> numerical operations
# os -> file and directory handling
# matplotlib -> plotting forecast graphs
# Prophet -> time series forecasting model
# sklearn.metrics -> evaluation metrics (MAE, RMSE)
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")   # suppress warning messages for cleaner output


# ------------------------------------------------------------
# Configuration Section
# ------------------------------------------------------------

# Path to dataset file
csv_path = "/Users/5212068/Downloads/sensor_data.csv"

# Output directory where results will be saved
out_dir = "prophet_tuning_results"

# Create output folders if they do not already exist
os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)


# ------------------------------------------------------------
# Load and Preprocess Dataset
# ------------------------------------------------------------

# Read CSV file
df = pd.read_csv(csv_path)

# Combine date and time columns into single datetime column
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')

# Remove rows with invalid datetime values and sort chronologically
df = df.dropna(subset=['datetime']).sort_values('datetime')

# Convert important columns to numeric format
# If invalid values exist, they will be converted to NaN
for col in ['temperature','humidity','light','voltage','epoch','moteid']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Get all unique mote IDs (sensor IDs)
mote_ids = np.sort(df['moteid'].unique())


# ------------------------------------------------------------
# Prophet Hyperparameter Grid
# ------------------------------------------------------------
# This grid defines different combinations of parameters
# that will be tested to find the best configuration
# for each sensor.

grid = [
  {
   'changepoint_prior_scale':0.01,      # very rigid trend
   'n_changepoints':25,                 # number of possible trend breakpoints
   'changepoint_range':0.90,            # allow changepoints in first 90% of data
   'seasonality_prior_scale':5.0,       # strength of seasonality component
   'seasonality_mode':'additive'        # additive seasonality
  },

  {
   'changepoint_prior_scale':0.1,       # moderate trend flexibility
   'n_changepoints':25,
   'changepoint_range':0.95,
   'seasonality_prior_scale':5.0,
   'seasonality_mode':'additive'
  },

  {
   'changepoint_prior_scale':0.5,       # flexible trend
   'n_changepoints':50,
   'changepoint_range':0.98,
   'seasonality_prior_scale':10.0,
   'seasonality_mode':'additive'
  },

  {
   'changepoint_prior_scale':1.0,       # very flexible trend
   'n_changepoints':50,
   'changepoint_range':0.98,
   'seasonality_prior_scale':10.0,
   'seasonality_mode':'multiplicative'  # multiplicative seasonality
  },
]

# List to store final results for all sensors
results = []


# ------------------------------------------------------------
# Function: Remove Extreme Spikes
# ------------------------------------------------------------
# Uses z-score method to detect extreme outliers
# and replaces them with interpolated values.

def remove_spikes(series, z_thresh=5.0):
    s = series.copy()
    z = (s - s.mean())/s.std()       # compute z-score
    s[z.abs() > z_thresh] = np.nan   # mark extreme values as NaN
    return s.interpolate(limit=6)    # fill small gaps using interpolation


# ------------------------------------------------------------
# Loop Through Each Sensor (Mote)
# ------------------------------------------------------------

for mote in mote_ids:

    # Filter data for current mote and set datetime as index
    mote_df = df[df['moteid']==mote].set_index('datetime')

    # Resample data to hourly frequency (mean aggregation)
    hourly = mote_df['temperature'].resample('H').mean()

    # Drop missing values
    hourly = hourly.dropna()

    # Skip sensors with insufficient data (less than 2 weeks)
    if len(hourly) < 24*14:
        print(f"Skipping mote {mote} (too little data: {len(hourly)})")
        continue


    # --------------------------------------------------------
    # Optional Data Cleaning
    # --------------------------------------------------------
    hourly_clean = remove_spikes(hourly, z_thresh=6.0)


    # --------------------------------------------------------
    # Train-Test Split
    # --------------------------------------------------------
    # Last 7 days used for testing
    test_hours = 24*7

    train = hourly_clean.iloc[:-test_hours]
    test  = hourly_clean.iloc[-test_hours:]


    # Initialize tracking variables for grid search
    best_score = np.inf
    best_cfg = None
    best_pred = None


    # --------------------------------------------------------
    # Hyperparameter Grid Search
    # --------------------------------------------------------
    for cfg in grid:
        try:
            # Initialize Prophet model with current configuration
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=cfg['changepoint_prior_scale'],
                n_changepoints=cfg['n_changepoints'],
                changepoint_range=cfg['changepoint_range'],
                seasonality_prior_scale=cfg['seasonality_prior_scale'],
                seasonality_mode=cfg['seasonality_mode'],
                interval_width=0.80
            )

            # Prepare training dataframe in Prophet format
            # Prophet requires columns named: ds (date) and y (target)
            train_df = train.reset_index().rename(
                columns={'datetime':'ds','temperature':'y'}
            )

            # Train model
            model.fit(train_df)

            # Create future dataframe for forecast horizon
            future = model.make_future_dataframe(periods=test_hours, freq='H')

            # Generate forecast
            fcst = model.predict(future)

            # Extract predictions corresponding to test period
            yhat = fcst[['ds','yhat']].set_index('ds') \
                   .iloc[-test_hours:]['yhat'].values

            # ------------------------------------------------
            # Model Evaluation
            # ------------------------------------------------
            mae = mean_absolute_error(test.values, yhat)
            rmse = np.sqrt(mean_squared_error(test.values, yhat))

            score = rmse  # choose RMSE as optimization metric

            # Keep configuration with lowest RMSE
            if score < best_score:
                best_score = score
                best_cfg = cfg.copy()
                best_pred = yhat

        except Exception as e:
            # Skip configuration if model fails
            continue


    # If no model worked for this sensor, skip it
    if best_cfg is None:
        print(f"No successful model for mote {mote}")
        continue


    # --------------------------------------------------------
    # Store Best Model Results
    # --------------------------------------------------------
    mae_final = mean_absolute_error(test.values, best_pred)
    rmse_final = np.sqrt(mean_squared_error(test.values, best_pred))

    results.append({
        'moteid': int(mote),
        'mae': mae_final,
        'rmse': rmse_final,
        'best_changepoint_prior_scale': best_cfg['changepoint_prior_scale'],
        'best_n_changepoints': best_cfg['n_changepoints'],
        'best_changepoint_range': best_cfg['changepoint_range'],
        'best_seasonality_prior_scale': best_cfg['seasonality_prior_scale'],
        'best_seasonality_mode': best_cfg['seasonality_mode']
    })


    # --------------------------------------------------------
    # Save Forecast Plot
    # --------------------------------------------------------

    # Show last 3 days of training data for context
    idx_train_display = train.index[-24*3:]

    plt.figure(figsize=(10,4))
    plt.plot(idx_train_display,
             train.loc[idx_train_display].values,
             label="Train (Last 3 Days)")

    plt.plot(test.index,
             test.values,
             label="Actual")

    plt.plot(test.index,
             best_pred,
             label="Forecast")

    plt.title(f"Tuned Prophet - Mote {mote} "
              f"(best cp_scale={best_cfg['changepoint_prior_scale']})")

    plt.legend()
    plt.tight_layout()

    # Save plot to output folder
    plt.savefig(os.path.join(out_dir,
                             "plots",
                             f"mote_{int(mote)}.png"))
    plt.close()

    print(f"Done mote {mote}: "
          f"MAE={mae_final:.3f} "
          f"RMSE={rmse_final:.3f} "
          f"cfg={best_cfg}")


# ------------------------------------------------------------
# Save All Performance Metrics to CSV
# ------------------------------------------------------------

pd.DataFrame(results).to_csv(
    os.path.join(out_dir,
                 "prophet_tuning_per_mote.csv"),
    index=False
)

print("All done. Results saved to:", out_dir)