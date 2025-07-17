import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

os.chdir(os.path.dirname(__file__))

dep = 'log_price'
indep = ['USDX.FX', 'VIX', 'cut', 'OPEC', 'CPI', 'DGS10', 'log_ret_lag']
indep = ['USDX.FX', 'OPEC', 'CPI', 'DGS10', 'log_ret_lag']
num_indep = len(indep)

show_coeff = 1

# 读取数据（你可以替换为实际文件路径或使用 StringIO 来测试）
file = 'merged_selected_data.csv'
df = pd.read_csv(file, encoding = 'gbk')

# 处理日期
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace = True)
df.asfreq('B')

# 计算对数收益率
df = df.dropna(subset = ['B.IPE'])
df['log_price'] = np.log(df['B.IPE']) 
df['log_ret'] = np.log(df['B.IPE']/df['B.IPE'].shift(1))

# 归中
df['USDX.FX'] = (df['USDX.FX'] - df['USDX.FX'].mean()) / df['USDX.FX'].std()*0.01
df['VIX'] = (df['VIX'] - df['VIX'].mean()) / df['VIX'].std()*0.01
df['CPI'] = (df['CPI'] - df['CPI'].mean()) / df['CPI'].std()*0.01
df['DGS10'] = (df['DGS10'] - df['DGS10'].mean()) / df['DGS10'].std()*0.01
df['OPEC'] = (df['OPEC'] - df['OPEC'].mean()) / df['OPEC'].std()*0.01
df['log_ret_lag'] = (df['log_ret'] - df['log_ret'].mean()) / df['log_ret'].std()*0.001
df['cut'] = df['cut']

# 向后补全
df[['OPEC', 'CPI']] = df[['OPEC', 'CPI']].fillna(method='ffill')


# 对齐
df = df[[dep] + indep]
for ind in indep:
    df[ind] = df[ind].shift(1)
df = df.dropna(subset = indep)

# 设置自变量和因变量
X = df[indep]
X = sm.add_constant(X)
y = df[dep]
data = pd.concat([X, y], axis=1).dropna()

# 拟合线性回归模型
model = sm.OLS(y, X).fit()
whole = model.summary()



import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# 设置滚动窗口大小（例如使用过去250天数据进行预测）
window_size = 250
gap = 50

# 准备空列表存储预测结果
predictions = []
true_values = []
dates = []
coefficient = [[],[],[],[],[],[],[],[],[],[],[],[]]#[] * num_indep

# 转换为 NumPy array 加快速度
const_indep = ['const'] + indep
for i in range(window_size, len(data) - 1 - gap):
    train_data = data.iloc[i - window_size:i]
    test_data = data.iloc[i + 1 + gap]

    X_train = train_data[const_indep]
    y_train = train_data[dep]
    
    model = sm.OLS(y_train, X_train).fit()

    X_test = test_data[const_indep]
    pred = model.predict(X_test)

    if pred.values[0] < 3 and False:
        print(model.summary())
        print(X_train)
    
    predictions.append(pred.values[0])
    #predictions.append(data.iloc[i][dep])
    true_values.append(test_data[dep])
    dates.append(data.index[i + 1])

    for i in range(num_indep + 1):
        coefficient[i].append(model.params[i])

# 转换为 pandas Series
pred_series = pd.Series(predictions, index=dates)
true_series = pd.Series(true_values, index=dates)

# 计算预测性能
r2 = r2_score(true_series, pred_series)
print(f"\nRolling Forecast R²: {r2:.4f}")

# 画图
plt.figure(figsize=(12, 6))

if show_coeff:
    for i in range(num_indep + 1):
        coef_series = pd.Series(coefficient[i], index=dates)
        plt.plot(coef_series.index, coef_series, label=f'coefficient {X.columns[i]}', linewidth=1)
else:
    plt.plot(true_series.index, true_series, label=f'True {dep}', linewidth=1)
    plt.plot(pred_series.index, pred_series, label=f'Predicted {dep}', linewidth=1)

plt.title("Rolling OLS Prediction vs. True Values")
plt.xlabel("Date")
plt.ylabel(dep)
plt.legend()
plt.tight_layout()
plt.show()

print(whole)