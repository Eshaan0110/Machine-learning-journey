# Prophet Time Series Forecasting

## Overview

This project explores **time series forecasting using the Prophet model**. Prophet is a forecasting library developed by Meta that is designed to handle time series data with strong seasonality and trend patterns.

The goal of this project is to understand how forecasting models can learn patterns from historical data and predict future values.

---

## What is Prophet?

Prophet is a forecasting model used for predicting future values in time series data such as:

* Website traffic
* Stock prices
* Sales forecasting
* Energy consumption
* Ride demand

The model decomposes time series data into three main components:

**Trend**
Long-term increase or decrease in the data.

**Seasonality**
Repeating patterns such as weekly or yearly cycles.

**Holidays / Events**
Special events that cause sudden changes in behavior.

The model can be expressed as:

y(t) = g(t) + s(t) + h(t) + ε

Where:

* g(t) = trend
* s(t) = seasonality
* h(t) = holiday effects
* ε = random noise

---

## Project Structure

Machine-learning-journey
│
├── prophet-learning
│   ├── data
│   │   └── website_traffic.csv
│   │
│   ├── results
│   │   ├── forecast.png
│   │   └── components.png
│   │
│   ├── src
│   │   └── forecast.py
│   │
│   └── README.md

---

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install prophet pandas matplotlib scikit-learn
```

---

## Running the Project

Navigate to the source folder:

```bash
cd prophet-learning/src
```

Run the forecasting script:

```bash
python forecast.py
```

This will generate forecast visualizations and save them in the `results` folder.

---

## Output

The model produces two visualizations:

**Forecast Plot**
Shows historical data along with predicted future values.

**Component Plot**
Breaks the prediction into:

* trend
* seasonal patterns

---

## Learning Outcomes

Through this project we learn:

* Time series preprocessing
* Forecasting with Prophet
* Trend and seasonality analysis
* Visualizing predictions

---

## Future Improvements

* Use larger real-world datasets
* Add anomaly detection
* Compare Prophet with ARIMA and LSTM models
