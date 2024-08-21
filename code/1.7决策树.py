import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 设置全局字体为中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 读取数据，指定编码为GBK
train_data = pd.read_csv('train2.csv', encoding='GBK')

# 只提取前1万行数据作为训练集，后1千行数据作为测试集
train_data_subset = train_data.head(11000)
train_data_train = train_data_subset.head(10000)
train_data_test = train_data_subset.tail(1000)

# 数据预处理
X_train = train_data_train.drop(columns=['id', '洪水概率'])
y_train = train_data_train['洪水概率']
X_test = train_data_test.drop(columns=['id', '洪水概率'])
y_test = train_data_test['洪水概率']

# 训练决策树模型
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# 计算训练集和测试集上的均方误差
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
train_mse = mean_squared_error(y_train, train_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

print(f"训练集均方误差: {train_mse}")
print(f"测试集均方误差: {test_mse}")

# 后剪枝操作
path = model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

models = []
for ccp_alpha in ccp_alphas:
    model = DecisionTreeRegressor(random_state=42, ccp_alpha=ccp_alpha)
    model.fit(X_train, y_train)
    models.append(model)

# 找出最佳的ccp_alpha
train_scores = [mean_squared_error(y_train, model.predict(X_train)) for model in models]
test_scores = [mean_squared_error(y_test, model.predict(X_test)) for model in models]

best_alpha = ccp_alphas[test_scores.index(min(test_scores))]
best_model = DecisionTreeRegressor(random_state=42, ccp_alpha=best_alpha)
best_model.fit(X_train, y_train)

print(f"最佳的 ccp_alpha: {best_alpha}")

# 提取特征重要性
feature_importances = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# 可视化特征重要性
plt.figure(figsize=(12, 8))
feature_importances.plot(kind='bar', color='skyblue')
plt.title('特征重要性')
plt.xlabel('特征')
plt.ylabel('重要性')
plt.show()

# 输出重要性较低的特征
low_importance_features = feature_importances[feature_importances < 0.01].index.tolist()
print("重要性较低的特征:", low_importance_features)

# 可视化决策树
plt.figure(figsize=(20, 10))
plot_tree(best_model, feature_names=X_train.columns, filled=True, rounded=True, fontsize=10)
plt.show()
