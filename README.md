# AAI530 Group 10 Data Analytics IoT Project

## 📌 Project Overview

This repository contains the final project submission for **AAI 530 – Data Analytics**.  
The objective of this project is to analyze sensor data from the **Intel Berkeley Research Lab**, perform exploratory data analysis (EDA), build forecasting models, and derive actionable insights from IoT time series patterns.

The dataset used is publicly available on **Kaggle**:

➡️ https://www.kaggle.com/datasets/divyansh22/intel-berkeley-research-lab-sensor-data

This project demonstrates data preprocessing, modeling, visualization, and performance evaluation.

---

## 📋 Dataset Description

The dataset consists of **sensor readings** from multiple nodes placed in the Intel Berkeley Research Lab. Each sensor records:

- Temperature
- Humidity
- Light intensity
- Voltage

along with other environmental measurements captured at regular time intervals.

The goal is to understand patterns, detect anomalies, and forecast relevant sensor signals over time.

---

## 🧠 Project Structure

The main analysis is contained in a Jupyter Notebook:

➡️ **Final_Project_AAI_530_Team10.ipynb**

The repository includes workflows for:

- Data ingestion
- Cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Time series modeling
- Model evaluation and comparison
- Visualization of forecast results

---

## 👥 Team Members & Responsibilities

| Team Member | Responsibilities |
|-------------|------------------|
| **Anugrah Rastogi** | Modelling, performance metrics calculation, forecast plots, and analysis |
| **Manoj Nair** | Data preprocessing, EDA, documentation, and Tableau dashboard creation |

---

## 🧩 Key Steps Performed

### 🛠 Data Preprocessing
- Handling missing values
- Time synchronization and formatting
- Stationarity checks
- Feature extraction for time series models

### 📊 Exploratory Data Analysis (EDA)
- Trend visualization over time
- Correlation analysis between sensors
- Seasonal patterns and behavior distributions

### 📈 Forecasting Models
We implemented the following models to forecast time-dependent sensor values:

- Time Series decomposition
- Prophet forecasting
- Traditional models (as applicable)

### 📉 Performance Metrics
We evaluated model performance using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

### 📈 Visualizations
Graphs and visualizations include:
- Time series plots
- Forecast vs Ground truth comparison
- Error distributions

---

## 📌 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Anugrahrastogi/AAI530-Group10-Data-analytics-IoT-Project.git