import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

# 设置全局字体为中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 读取数据，指定编码为GBK
train_data = pd.read_csv('train2.csv', encoding='GBK')

# 只提取前800000行数据进行分析
train_data = train_data.head(1000)

# 数据预处理
train_data = train_data.dropna()  # 删除缺失值

# 提取特征和目标变量
X = train_data.drop(columns=['id', '洪水概率'])
y = train_data['洪水概率']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 构建神经网络模型
model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)

# 训练模型
model.fit(X_train_scaled, y_train)

# 预测
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# 模型评估
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"训练集均方误差: {train_mse}")
print(f"测试集均方误差: {test_mse}")
print(f"训练集R^2: {train_r2}")
print(f"测试集R^2: {test_r2}")

# 可视化训练过程中的损失变化
plt.figure(figsize=(12, 6))
plt.plot(model.loss_curve_, label='训练损失')
plt.title('训练损失')
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.legend()
plt.show()

# 可视化预测结果的分布图和折线图
plt.figure(figsize=(12, 6))
plt.hist(y_pred_test, bins=50, alpha=0.6, color='g', label='预测洪水概率分布')
plt.xlabel('洪水概率')
plt.ylabel('频率')
plt.title('测试集预测洪水概率分布图')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='实际值', color='blue')
plt.plot(y_pred_test, label='预测值', color='red')
plt.xlabel('样本点')
plt.ylabel('洪水概率')
plt.title('测试集洪水概率预测折线图')
plt.legend()
plt.show()
