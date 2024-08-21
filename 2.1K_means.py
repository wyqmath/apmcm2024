import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

# 设置全局字体为中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 读取数据，指定编码为GBK
file_path = 'train2.csv'
train_data = pd.read_csv(file_path, encoding='GBK')

# 提取特征和标签
features = train_data.drop(columns=['id', '洪水概率'])
labels = train_data['洪水概率']

# 数据标准化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# KMeans聚类
kmeans = KMeans(n_clusters=3, random_state=42)
train_data['风险类别'] = kmeans.fit_predict(scaled_features)

# 分析不同类别的指标特征
cluster_means = train_data.groupby('风险类别').mean()

# 绘制不同类别的指标特征
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(20, 20))
fig.suptitle('特征的聚类均值', fontsize=20)

colors = sns.color_palette("husl", 3)  # 使用HUSL颜色空间生成不同的颜色
for i, column in enumerate(features.columns):
    ax = axes[i // 4, i % 4]
    cluster_means[column].plot(kind='bar', ax=ax, color=colors)
    ax.set_title(column)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 计算指标权重（使用随机森林）
rf = RandomForestClassifier(random_state=42)
rf.fit(scaled_features, train_data['风险类别'])
feature_importances = rf.feature_importances_

# 绘制特征重要性（条形图）
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances, y=features.columns, palette="viridis")
plt.title('要素重要性（条形图）')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# 绘制特征重要性（饼图）
plt.figure(figsize=(12, 8))
plt.pie(feature_importances, labels=features.columns, autopct='%1.1f%%', colors=sns.color_palette("viridis", len(features.columns)))
plt.title('指标重要性（饼图）')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# 建立预警评价模型（使用选定的重要特征）
important_features = [features.columns[i] for i in np.argsort(feature_importances)[-5:]]
X_important = train_data[important_features]

# 训练模型
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_important, train_data['风险类别'])

# 模型灵敏度分析（通过调整参数或移除特征）
# 评估模型性能
predictions = rf_model.predict(X_important)
print(classification_report(train_data['风险类别'], predictions))

# 灵敏度分析（移除一个特征）
for feature in important_features:
    X_temp = X_important.drop(columns=[feature])
    temp_model = RandomForestClassifier(random_state=42)
    temp_model.fit(X_temp, train_data['风险类别'])
    temp_predictions = temp_model.predict(X_temp)
    print(f'Removed feature: {feature}')
    print(classification_report(train_data['风险类别'], temp_predictions))

# 可视化结果
# 绘制风险类别的分布直方图
plt.figure(figsize=(12, 8))
sns.histplot(train_data['风险类别'], bins=3, kde=False, palette="viridis")
plt.title('风险类别分布')
plt.xlabel('Risk Category')
plt.ylabel('Frequency')
plt.show()



# 绘制重要特征的箱线图
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))
fig.suptitle('Box Plots of Important Features by Risk Category', fontsize=20)

for i, feature in enumerate(important_features):
    sns.boxplot(x=train_data['风险类别'], y=train_data[feature], ax=axes[i], palette="viridis")
    axes[i].set_title(feature)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 绘制重要特征的箱线图
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))
fig.suptitle('Box Plots of Important Features by Risk Category', fontsize=20)

for i, feature in enumerate(important_features):
    sns.boxplot(x=train_data['风险类别'], y=train_data[feature], ax=axes[i])
    axes[i].set_title(feature)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# 绘制重要特征的雷达图
def plot_radar(data, labels, title):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    data = np.concatenate((data, [data[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, data, color='red', alpha=0.25)
    ax.plot(angles, data, color='red', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    plt.title(title, size=20, color='red', y=1.1)
    plt.show()


# 计算不同风险类别的重要特征均值
mean_values = train_data.groupby('风险类别')[important_features].mean()

# 绘制雷达图
for i, risk in enumerate(mean_values.index):
    data = mean_values.loc[risk].values
    labels = mean_values.columns.tolist()
    plot_radar(data, labels, f'{risk} 风险类别的重要特征雷达图')

print("绘图完成！")
