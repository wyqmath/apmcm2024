import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 设置全局字体为中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 读取数据，指定编码为GBK
train_data = pd.read_csv('train2.csv', encoding='GBK')

# 相关性分析
correlation_matrix = train_data.corr()
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Spearman相关性分析热力图')
plt.show()

# 提取与洪水发生概率相关性最大的指标
target = '洪水概率'
correlation_with_target = correlation_matrix[target].sort_values(ascending=False)
print("与洪水发生概率相关性最大的指标：")
print(correlation_with_target)

# 特征重要性分析
X = train_data.drop(columns=['id', '洪水概率'])
y = train_data['洪水概率']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 提取指标重要性
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# 可视化指标重要性
plt.figure(figsize=(12, 8))
feature_importances.plot(kind='bar', color='skyblue')
plt.title('指标重要性')
plt.xlabel('指标')
plt.ylabel('重要性')
plt.show()

# 输出重要性较低的特征
low_importance_features = feature_importances[feature_importances < 0.01].index.tolist()
print("重要性较低的指标:", low_importance_features)
