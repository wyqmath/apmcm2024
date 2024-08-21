import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest, norm, probplot
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

#加载数据
file_path = 'train.xlsx'
train_data = pd.read_excel(file_path)

# 获取指标名称（除去'id'和'洪水概率'）
feature_columns = train_data.columns.difference(['id', '洪水概率'])

# 创建图形对象
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(20, 20))
fig.suptitle('线性回归散点图', fontsize=20)

# 绘制散点图与回归线
for i, column in enumerate(feature_columns):
    ax = axes[i // 4, i % 4]
    sns.regplot(x=train_data[column], y=train_data['洪水概率'], ax=ax, scatter_kws={'s': 10}, line_kws={'color': 'red'})
    ax.set_title(column)
    ax.set_xlabel(column)
    ax.set_ylabel('洪水概率')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 创建直方图和密度图对象
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(20, 20))
fig.suptitle('直方图和密度图', fontsize=20)

for i, column in enumerate(feature_columns):
    ax = axes[i // 4, i % 4]
    sns.histplot(train_data[column], kde=True, ax=ax, color='skyblue', edgecolor='black')
    ax.set_title(column)
    ax.set_xlabel(column)
    ax.set_ylabel('频率')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
