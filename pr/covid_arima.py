import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Sample COVID-like data simulation
date_rng = pd.date_range(start="2020-01-01", periods=50, freq="W")
cases = np.random.randint(100, 1000, size=len(date_rng))

df = pd.DataFrame({"Date": date_rng, "Cases": cases})
df.set_index("Date", inplace=True)

# Plot
df.plot(title="COVID-19 Weekly Cases")
plt.ylabel("Cases")
plt.show()

# ARIMA model
model = ARIMA(df["Cases"], order=(2, 1, 2))
model_fit = model.fit()
print(model_fit.summary())

# Forecast next 10 weeks
forecast = model_fit.forecast(steps=10)
plt.plot(df.index, df["Cases"], label="Actual")
plt.plot(
    pd.date_range(df.index[-1], periods=10, freq="W"),
    forecast,
    label="Forecast",
    linestyle="--",
)
plt.title("ARIMA Forecast")
plt.legend()
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Simulate COVID-like weekly data
# -------------------------------
np.random.seed(42)
date_rng = pd.date_range(start="2020-01-01", periods=50, freq="W")
cases = np.random.randint(100, 1000, size=len(date_rng))

df = pd.DataFrame({"Date": date_rng, "Cases": cases})
df.set_index("Date", inplace=True)

plt.plot(df.index, df["Cases"], label="Actual Cases")
plt.title("COVID-19 Weekly Cases")
plt.ylabel("Cases")
plt.legend()
plt.show()

# -------------------------------
# 2️⃣ Implement ARIMA(1,1,1) manually
# -------------------------------


def difference(series):
    """Differencing to make data stationary"""
    return np.diff(series)


def inverse_difference(original, diff_series):
    """Revert differenced series back to original scale"""
    result = [original[0]]
    for value in diff_series:
        result.append(result[-1] + value)
    return np.array(result)


def manual_arima(series, p=1, d=1, q=1, forecast_steps=10):
    """Simplified ARIMA(p,d,q)"""
    # Step 1: Differencing (I part)
    diff_series = difference(series)

    # Initialize parameters (for AR(1), MA(1))
    phi = 0.7  # AR coefficient
    theta = 0.6  # MA coefficient
    mu = np.mean(diff_series)  # Mean of differenced data

    errors = [0]
    predictions = []

    # Step 2: Fit ARMA(1,1) manually
    for t in range(1, len(diff_series)):
        pred = mu + phi * diff_series[t - 1] + theta * errors[-1]
        error = diff_series[t] - pred
        errors.append(error)
        predictions.append(pred)

    # Step 3: Forecast future steps
    future_preds = []
    last_diff = diff_series[-1]
    last_err = errors[-1]
    for _ in range(forecast_steps):
        next_pred = mu + phi * last_diff + theta * last_err
        future_preds.append(next_pred)
        last_err = 0  # assume no error for future
        last_diff = next_pred

    # Step 4: Combine results and invert differencing
    all_diffs = np.concatenate([diff_series[:1], predictions, future_preds])
    forecast_full = inverse_difference(series, all_diffs)

    return forecast_full, len(series)


# Run manual ARIMA
forecast_full, split_point = manual_arima(
    df["Cases"].values, p=1, d=1, q=1, forecast_steps=10
)

# -------------------------------
# 3️⃣ Visualization
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(df.index, df["Cases"], label="Actual")
future_dates = pd.date_range(df.index[-1], periods=11, freq="W")[1:]
plt.plot(future_dates, forecast_full[split_point:], "--", label="Forecast")
plt.title("Manual ARIMA(1,1,1) Forecast")
plt.ylabel("Cases")
plt.legend()
plt.show()
