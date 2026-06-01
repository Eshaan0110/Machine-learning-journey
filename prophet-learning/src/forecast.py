import sys
import os
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

DATA_PATH = "../data/website_traffic.csv"
RESULTS_DIR = "../results"


def load_data(path):
    if not os.path.exists(path):
        print(f"Error: data file not found at {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Forecast plot
    fig1 = model.plot(forecast)
    fig1.savefig(os.path.join(RESULTS_DIR, "forecast.png"))
    plt.close(fig1)
    print("Saved forecast.png")

    # Component plot (trend + seasonality breakdown)
    fig2 = model.plot_components(forecast)
    fig2.savefig(os.path.join(RESULTS_DIR, "components.png"))
    plt.close(fig2)
    print("Saved components.png")


if __name__ == "__main__":
    main()
