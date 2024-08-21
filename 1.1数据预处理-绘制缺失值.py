import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局字体为中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 读取数据，指定编码为GBK
train_data = pd.read_csv('train.csv', encoding='GBK')

# 检查缺失值
missing_values = train_data.isnull().sum()
missing_values = missing_values[missing_values > 0]

# 可视化缺失值
plt.figure(figsize=(12, 8))
plt.bar(missing_values.index, missing_values.values, color='skyblue')
plt.xticks(rotation=90)
plt.title('数据预处理-缺失值查找')
plt.xlabel('特征')
plt.ylabel('缺失值数量')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 绘制箱线图检查异常值
plt.figure(figsize=(16, 24))
num_cols = 1  # 每行显示的图表数量
features = [col for col in train_data.columns if col != 'id' and col != 'flood_probability']

for i, col in enumerate(features):
    plt.subplot(len(features) // num_cols + 1, num_cols, i + 1)  # 创建子图
    plt.boxplot(train_data[col].dropna(), vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen'))  # 去除缺失值后绘制箱线图
    plt.title(col)
    plt.xlabel('取值范围')
    plt.ylabel(col)
plt.tight_layout()
plt.show()

# 输出每个向量的取值范围区间
for col in features:
    col_min = train_data[col].min()
    col_max = train_data[col].max()
    print(f'{col}: 范围 [{col_min}, {col_max}]')

# 绘制属性的分布图
plt.figure(figsize=(20, 30))
for i, col in enumerate(features):
    plt.subplot(len(features) // num_cols + 1, num_cols, i + 1)
    sns.histplot(train_data[col].dropna(), kde=True, color='skyblue')
    plt.title(f'{col} 分布图')
    plt.xlabel(col)
    plt.ylabel('频率')
plt.tight_layout()
plt.show()
