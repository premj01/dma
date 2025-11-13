# 📘 Simple Time Series Analysis and Forecasting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Create sample dataset
months = pd.date_range(start="2022-01", periods=24, freq="M")
sales = [
    200,
    220,
    250,
    270,
    300,
    330,
    360,
    400,
    420,
    450,
    470,
    500,
    520,
    540,
    560,
    600,
    620,
    640,
    660,
    680,
    700,
    720,
    740,
    760,
]
data = pd.DataFrame({"Month": months, "Sales": sales})

# Step 2: Descriptive analytics
print("📊 Data Summary:")
print(data.describe())

plt.plot(data["Month"], data["Sales"], marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()

# Step 3: Forecasting model
model = ARIMA(data["Sales"], order=(1, 1, 1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=6)

# Step 4: Plot forecast
future_months = pd.date_range(data["Month"].iloc[-1], periods=6, freq="M")
plt.plot(data["Month"], data["Sales"], label="Actual")
plt.plot(future_months, forecast, color="red", label="Forecast")
plt.title("Sales Forecast (Next 6 Months)")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.show()

print("🔮 Forecasted Sales:")
print(pd.DataFrame({"Month": future_months, "Predicted_Sales": forecast}))
