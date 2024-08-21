import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

# 设置全局字体为中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 读取数据，指定编码为GBK
train_data = pd.read_csv('train2.csv', encoding='GBK')

# 只提取前80万行数据进行分析
train_data = train_data.head(800000)

# 因子分析前需要标准化数据
scaler = StandardScaler()
X = train_data.drop(columns=['id', '洪水概率'])
X_scaled = scaler.fit_transform(X)

# 因子分析，选择适当的因子数量
fa = FactorAnalysis(n_components=5, random_state=42)
X_fa = fa.fit_transform(X_scaled)

# 因子载荷矩阵
factor_loadings = pd.DataFrame(fa.components_.T, index=X.columns, columns=[f'因子{i+1}' for i in range(fa.n_components)])

# 可视化因子载荷矩阵
plt.figure(figsize=(16, 8))
sns.heatmap(factor_loadings, annot=True, cmap='coolwarm', center=0)
plt.title('因子载荷矩阵')
plt.show()

# 输出因子载荷矩阵
print("因子载荷矩阵：")
print(factor_loadings)

# 提取前10个最重要的特征指标
top_features = factor_loadings.abs().max(axis=1).sort_values(ascending=False).head(10)

# 可视化最重要的前10个特征指标
plt.figure(figsize=(12, 8))
top_features.plot(kind='bar', color='skyblue')
plt.title('最重要的前10个特征指标')
plt.xlabel('特征')
plt.ylabel('重要性')
plt.show()

# 输出最重要的前10个特征指标
print("最重要的前10个特征指标：")
print(top_features)
