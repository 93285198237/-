import os
os.chdir(os.path.dirname(__file__))

import pandas as pd
import numpy as np
from functools import reduce
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# === Step 1: 导入原始数据 ===
merged = pd.read_csv("merged_selected_data.csv")

# 处理日期
merged['Date'] = pd.to_datetime(merged['Date'])
merged.set_index('Date', inplace = True)
merged.asfreq('B')

# 计算对数收益率
merged = merged.dropna(subset = ['B.IPE'])
merged['log_price'] = np.log(merged['B.IPE']) 
merged['log_ret_BIPE'] = np.log(merged['B.IPE']/merged['B.IPE'].shift(1))


# === Step 2: Rolling GARCH prediction ===
scale = 100
garch_window = 250
train_window = 260
log_ret = merged["log_ret_BIPE"].dropna()
vol_preds, vol_realized, vol_dates = [], [], []
print(train_window, len(log_ret) - 1)
for t in range(train_window, len(log_ret) - 1):
    y_train = log_ret.iloc[t - garch_window:t] * scale
    model = arch_model(y_train, mean="AR", lags=0, vol="GARCH", p=1, q=1, dist="t")
    res = model.fit(disp="off")
    sigma_pred = np.sqrt(res.forecast(horizon=1).variance.values[-1, 0]) / scale
    sigma_real = log_ret.iloc[t + 1 - 4:t + 2].std()
    vol_preds.append(sigma_pred)
    vol_realized.append(sigma_real)
    vol_dates.append(log_ret.index[t + 1])

df_vol_compare = pd.DataFrame({
    "garch_vol_pred": vol_preds,
    "vol_real": vol_realized
}, index=vol_dates)

r2_vol = r2_score(df_vol_compare["vol_real"], df_vol_compare["garch_vol_pred"])

# === Save volatility plot ===
plt.figure(figsize=(12, 5))
plt.plot(df_vol_compare.index, df_vol_compare["vol_real"], label="Realized (5-day STD)", linewidth=1)
plt.plot(df_vol_compare.index, df_vol_compare["garch_vol_pred"], label="Predicted (GARCH)", linewidth=1)
plt.title("Volatility Prediction: GARCH vs Realized")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"garch_vol_prediction(R2={r2_vol:.4f}).png", dpi=300)
plt.close()


