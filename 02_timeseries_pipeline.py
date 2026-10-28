"""
Standalone Time Series Pipeline – Amazon Customer Reviews (100-pt version)
Resampling, multi-test stationarity (ADF+KPSS), seasonal decomposition,
ACF/PACF, multi-model ARIMA comparison, forecasting.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# 1. Load & prepare
# ------------------------------------------------------------------
df = pd.read_csv('amazon_reviews_lab.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Earliest review: {df['timestamp'].min()}")
print(f"Latest review:  {df['timestamp'].max()}")

df = df.set_index('timestamp')
monthly_review_count = df.resample('M').size()
monthly_avg_rating   = df['rating'].resample('M').mean()

# ------------------------------------------------------------------
# 2. Trend plots
# ------------------------------------------------------------------
plt.figure(figsize=(12, 5))
monthly_review_count.plot(color='blue')
plt.title('Monthly Review Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Reviews')
plt.grid(True)
plt.tight_layout()
plt.show()

rolling = monthly_avg_rating.rolling(window=3).mean()
plt.figure(figsize=(12, 5))
monthly_avg_rating.plot(label='Original Monthly Avg', color='lightgray', marker='o')
rolling.plot(label='3-Month Rolling Avg', color='red', linewidth=2)
plt.title('Average Monthly Rating Over Time')
plt.xlabel('Date')
plt.ylabel('Average Rating (1-5)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 3. Stationarity (ADF + KPSS + seasonal decomposition)
# ------------------------------------------------------------------
print("=== ADF Test (Original) ===")
adf_orig = adfuller(monthly_review_count)
print(f"ADF Statistic: {adf_orig[0]:.4f} | p-value: {adf_orig[1]:.4f}")

print("\n=== KPSS Test (Original) ===")
kpss_orig = kpss(monthly_review_count, regression='c')
print(f"KPSS Statistic: {kpss_orig[0]:.4f} | p-value: {kpss_orig[1]:.4f}")

decomp = seasonal_decompose(monthly_review_count, model='additive', period=12)
fig = decomp.plot()
fig.set_size_inches(12, 8)
plt.suptitle("Seasonal Decomposition of Monthly Review Volume", y=1.02)
plt.tight_layout()
plt.show()

monthly_diff = monthly_review_count.diff().dropna()
print("\n=== ADF Test (Differenced) ===")
adf_diff = adfuller(monthly_diff)
print(f"ADF Statistic: {adf_diff[0]:.4f} | p-value: {adf_diff[1]:.4f}")

plt.figure(figsize=(12, 4))
monthly_diff.plot(color='green')
plt.title('Stationary Series: First-Order Differenced Monthly Reviews')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.ylabel('Differenced Count')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 4. ACF / PACF + multi-model ARIMA comparison
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(monthly_diff, ax=axes[0], lags=20)
plot_pacf(monthly_diff, ax=axes[1], lags=20)
axes[0].set_title("ACF – Differenced Series")
axes[1].set_title("PACF – Differenced Series")
plt.tight_layout()
plt.show()

train_size = int(len(monthly_review_count) * 0.8)
train = monthly_review_count.iloc[:train_size]
test  = monthly_review_count.iloc[train_size:]

candidates = {
    "ARIMA(1,1,1)": (1, 1, 1),
    "ARIMA(2,1,1)": (2, 1, 1),
    "ARIMA(1,1,2)": (1, 1, 2),
}

results = {}
print("\nModel Comparison:")
print(f"{'Model':<15} {'AIC':>10} {'BIC':>10} {'RMSE':>10}")
print("-" * 50)

for name, order in candidates.items():
    fit = ARIMA(train, order=order).fit()
    pred = fit.forecast(steps=len(test))
    rmse = np.sqrt(mean_squared_error(test, pred))
    results[name] = {"fit": fit, "pred": pred, "AIC": fit.aic, "BIC": fit.bic, "RMSE": rmse}
    print(f"{name:<15} {fit.aic:>10.1f} {fit.bic:>10.1f} {rmse:>10.2f}")

best_name = min(results, key=lambda k: results[k]["RMSE"])
print(f"\nBest model by RMSE: {best_name}")

best = results[best_name]
print(best["fit"].summary().tables[0])

mae = mean_absolute_error(test, best["pred"])
print(f'\nForecast Evaluation -> RMSE: {best["RMSE"]:.2f} | MAE: {mae:.2f}')

plt.figure(figsize=(12, 5))
train.plot(label='Training Data')
test.plot(label='Actual Test Data', color='orange')
best["pred"].plot(label=f'{best_name} Forecast', color='red', linestyle='--')
plt.title('ARIMA Forecasting: Expected vs. Actual Review Volume')
plt.legend()
plt.tight_layout()
plt.show()
