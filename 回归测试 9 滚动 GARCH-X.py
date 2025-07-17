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
merged["VIX_lag1"] = merged["VIX"].shift(1)
merged['log_price'] = np.log(merged['B.IPE']) 
merged['log_ret_BIPE'] = np.log(merged['B.IPE']/merged['B.IPE'].shift(1))

# === Step 2: Rolling GARCH-X prediction ===
scale = 1000
garch_window = 250
train_window = 260
v_window = 3
log_ret = merged["log_ret_BIPE"]
X_all = merged[["VIX_lag1"]]
combined = pd.concat([log_ret, X_all], axis=1).dropna()
log_ret = combined["log_ret_BIPE"]
X_all = combined[["VIX_lag1"]]

vol_preds, vol_realized, vol_dates = [], [], []

for t in range(train_window, len(log_ret) - 1):
    y_train = log_ret.iloc[t - garch_window:t] * scale
    x_train = X_all.iloc[t - garch_window:t]

    data = pd.concat([y_train, x_train], axis=1).dropna()
    y_clean = data.iloc[:, 0]
    x_clean = data.iloc[:, 1:].copy()

    if len(y_clean) < 50:
        continue

    model = arch_model(
    y_clean,
    mean="LS",        # ✅ 用线性回归均值项以启用 x
    vol="GARCH",
    p=1,
    q=1,
    dist="t",
    x=x_clean
    )
    res = model.fit(disp="off")

    x_forecast = X_all.iloc[t:t+1]  # ✅ 保持为 DataFrame，保留列名
    sigma_pred = np.sqrt(res.forecast(horizon=1, x=x_forecast).variance.values[-1, 0]) / scale
    sigma_real = log_ret.iloc[t + 1 - 4:t + 2].std()
    vol_preds.append(sigma_pred)
    vol_realized.append(sigma_real)
    vol_dates.append(log_ret.index[t])

    b = - train_window + len(log_ret) - 1
    pct = 100 * (t-train_window+1) //  b
    if  (t-train_window) %  (b//10) == 0:
        print(f"GARCHX {pct}% done...")

df_vol_compare = pd.DataFrame({
    "garch_vol_pred": vol_preds,
    "vol_real": vol_realized
}, index=vol_dates)

r2_vol = r2_score(df_vol_compare["vol_real"], df_vol_compare["garch_vol_pred"])

plt.figure(figsize=(12, 5))
plt.plot(df_vol_compare.index, df_vol_compare["vol_real"], label="Realized (5-day STD)", linewidth=1)
plt.plot(df_vol_compare.index, df_vol_compare["garch_vol_pred"], label="Predicted (GARCH-X)", linewidth=1)
plt.title("Volatility Prediction: GARCH-X vs Realized")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"garchx_vol_prediction(R2={r2_vol:.4f}).png", dpi=300)
plt.close()

