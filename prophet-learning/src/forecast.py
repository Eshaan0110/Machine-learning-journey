import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv("../data/website_traffic.csv")

print("Columns in dataset:", df.columns)

# rename columns safely
df.columns = ["ds", "y"]

# convert to datetime
df["ds"] = pd.to_datetime(df["ds"])

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=30)

forecast = model.predict(future)

model.plot(forecast)
plt.savefig("../results/forecast.png")
plt.show()