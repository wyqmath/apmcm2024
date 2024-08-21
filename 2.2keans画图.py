import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
from mpl_toolkits.mplot3d import Axes3D

# 设置全局字体为中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 读取数据，指定编码为GBK
train_data = pd.read_csv('train2.csv', encoding='GBK')

# 只提取前1万行数据进行分析
train_data = train_data.head(10000)

# 聚类分析
X = train_data.drop(columns=['id', '洪水概率'])
y = train_data['洪水概率']

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用K-means聚类
kmeans = KMeans(n_clusters=3, random_state=42)
train_data['风险类别'] = kmeans.fit_predict(X_scaled)

# 使用PCA将数据降维到2维，以便可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 将PCA结果添加到数据集中
train_data['PCA1'] = X_pca[:, 0]
train_data['PCA2'] = X_pca[:, 1]

# 映射风险类别
risk_categories = {0: '低风险', 1: '中风险', 2: '高风险'}
train_data['风险类别'] = train_data['风险类别'].map(risk_categories)

# 可视化聚类结果
plt.figure(figsize=(16, 12))
sns.scatterplot(x='PCA1', y='PCA2', hue='风险类别', palette='viridis', data=train_data, s=100, alpha=0.7)
plt.title('K-means 聚类结果')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='风险类别', loc='upper right')
plt.grid(True)
plt.show()

# 3D 可视化
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)
train_data['PCA3'] = X_pca_3d[:, 2]
scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=train_data['风险类别'].apply(lambda x: {'低风险': 0, '中风险': 1, '高风险': 2}[x]), cmap='viridis', s=100, alpha=0.7)
ax.set_title('K-means 聚类结果 (3D)')
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
legend1 = ax.legend(*scatter.legend_elements(), title="风险类别", loc='upper right')
ax.add_artist(legend1)
plt.show()




