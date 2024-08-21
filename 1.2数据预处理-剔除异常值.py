import pandas as pd
import matplotlib.pyplot as plt

# 设置全局字体为中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 读取数据，指定编码为GBK
train_data = pd.read_csv('train.csv', encoding='GBK')

# 统计每个特征中为18的值的数量
for col in train_data.columns:
    if col != 'id' and col != 'flood_probability':
        num_18 = (train_data[col] == 18).sum()
        total = len(train_data[col])
        print(f'{col}: 18的数量 {num_18}, 占比 {num_18/total:.2%}')


# 删除特征>10的数据
import pandas as pd
import matplotlib.pyplot as plt

# 设置全局字体为中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 读取数据，指定编码为GBK
train_data = pd.read_csv('train1.csv', encoding='GBK')

# 统计每个特征中为18的值的数量
for col in train_data.columns:
    if col != 'id' and col != 'flood_probability':
        num_18 = (train_data[col] == 18).sum()
        total = len(train_data[col])
        print(f'{col}: 18的数量 {num_18}, 占比 {num_18/total:.2%}')

# 删除特征属性值大于10的所有对应的行
for col in train_data.columns:
    if col != 'id' and col != 'flood_probability':
        train_data = train_data[train_data[col] <= 10]

# 检查删除后的数据集大小
print(f'删除后的数据集大小: {train_data.shape}')

# 保存处理后的数据
train_data.to_csv('train2.csv', index=False, encoding='GBK')

# 查看处理后的数据
print(train_data.head())
