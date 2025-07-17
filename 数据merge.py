import pandas as pd
import os
from functools import reduce

# 设置路径为脚本所在路径（可选）
os.chdir(os.path.dirname(__file__))

# === 读取数据 ===
futures = pd.read_csv("_期货数据.csv")
bdi_usdx = pd.read_csv("BDI AND USDX.csv")
vix = pd.read_csv("VIX.csv")
opec_dummy = pd.read_csv("opec_cuts_binary_2020_2025.csv")
opec_output = pd.read_csv(f'OPEC_output_cleaned.csv')
CPI = pd.read_csv('CPI.csv')
DGS10 = pd.read_csv('DGS10.csv')

# === 统一日期格式 ===
futures["Date"] = pd.to_datetime(futures["Date"])
bdi_usdx["Date"] = pd.to_datetime(bdi_usdx["Date"])
vix["Date"] = pd.to_datetime(vix["Date"])
opec_dummy["Date"] = pd.to_datetime(opec_dummy["Date"])
opec_output["Date"] = pd.to_datetime(opec_output["Date"])
CPI["Date"] = pd.to_datetime(CPI["Date"])
DGS10["Date"] = pd.to_datetime(DGS10["Date"])

# === 选取需要的列 ===
futures = futures[["Date", "B.IPE"]]
bdi_usdx = bdi_usdx[["Date", "USDX.FX"]]
vix = vix[["Date", "VIX"]]
opec_dummy = opec_dummy[["Date", "cut"]]
opec_output = opec_output[["Date", "OPEC"]]
CPI = CPI[["Date", "CPI"]]
DGS10 = DGS10[["Date", "DGS10"]]

# === 按日期逐个外连接合并 ===
dfs = [futures, bdi_usdx, vix, opec_dummy, opec_output, CPI, DGS10]
merged = reduce(lambda left, right: pd.merge(left, right, on="Date", how="outer"), dfs)
merged = merged.sort_values("Date").reset_index(drop=True)
print(merged.head())

# === 保存最终文件 ===
merged.to_csv("merged_selected_data.csv", index=False)
print("已按日期合并，文件为 merged_selected_data.csv")

