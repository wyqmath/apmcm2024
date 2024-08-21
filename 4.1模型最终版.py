import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from scipy.stats import norm, shapiro

# 设置全局字体为中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 读取训练数据，指定编码为GBK
train_data = pd.read_csv('train2.csv', encoding='GBK')

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
model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, alpha=0.001, random_state=42)  # 添加alpha参数进行L2正则化

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

# 读取测试数据
test_data = pd.read_csv('test.csv', encoding='GBK')

# 提取测试数据中的特征，去除洪水概率列
X_test_final = test_data.drop(columns=['id', '洪水概率'], errors='ignore')  # 确保没有洪水概率列

# 标准化测试数据
X_test_final_scaled = scaler.transform(X_test_final)

# 预测测试数据中的洪水概率
y_pred_test_final = model.predict(X_test_final_scaled)

# 将预测结果保存到submit.csv中
submit = pd.read_csv('submit.csv', encoding='GBK')
submit['洪水概率'] = y_pred_test_final
submit.to_csv('submit.csv', index=False)

print("预测结果已成功保存到submit.csv中")

# 可视化测试数据预测结果的分布图
plt.figure(figsize=(12, 6))
sns.histplot(y_pred_test_final, bins=50, kde=True, color='g', label='预测洪水概率分布')
plt.xlabel('洪水概率')
plt.ylabel('频率')
plt.title('测试集预测洪水概率分布图')
plt.legend()
plt.show()

# 可视化测试数据预测结果的折线图
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(y_pred_test_final)), y_pred_test_final, label='预测值', color='red')
plt.xlabel('样本点')
plt.ylabel('洪水概率')
plt.title('测试集洪水概率预测折线图')
plt.legend()
plt.show()

# 分析预测结果的分布是否服从正态分布
plt.figure(figsize=(12, 6))
sns.histplot(y_pred_test_final, bins=50, kde=True, stat="density", color='g', label='预测洪水概率分布')
mean = np.mean(y_pred_test_final)
std_dev = np.std(y_pred_test_final)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean, std_dev)
plt.plot(x, p, 'k', linewidth=2)
title = "预测洪水概率分布拟合正态分布\n均值 = %.2f, 标准差 = %.2f" % (mean, std_dev)
plt.title(title)
plt.xlabel('洪水概率')
plt.ylabel('频率')
plt.legend()
plt.show()

# Shapiro-Wilk 正态性检验
stat, p = shapiro(y_pred_test_final)
print('Shapiro-Wilk 正态性检验统计量=%.3f, p值=%.3f' % (stat, p))

# 判断是否服从正态分布
alpha = 0.05
if p > alpha:
    print('样本服从正态分布 (接受 H0)')
else:
    print('样本不服从正态分布 (拒绝 H0)')
