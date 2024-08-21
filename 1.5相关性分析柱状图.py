import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局字体为中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 数据
correlation_data = {
    '特征': ['基础设施恶化', '地形排水', '季风强度', '大坝质量', '淤积', '河流管理', '人口得分', '滑坡', '气候变化', '森林砍伐',
             '无效防灾', '农业实践', '湿地损失', '流域', '规划不足', '政策因素', '城市化', '侵蚀', '排水系统', '海岸脆弱性'],
    '相关性': [0.192852, 0.191362, 0.191353, 0.189720, 0.189705, 0.189569, 0.188808, 0.187898, 0.187465, 0.187441,
               0.186922, 0.186685, 0.186139, 0.185143, 0.184784, 0.184480, 0.183468, 0.182404, 0.180169, 0.179923]
}

df = pd.DataFrame(correlation_data)

# 排序
df = df.sort_values(by='相关性', ascending=False)

# 绘制柱状图
plt.figure(figsize=(12, 8))
sns.barplot(x='相关性', y='特征', data=df, palette='viridis')
plt.title('与洪水发生概率相关性最大的特征', fontsize=16)
plt.xlabel('相关性', fontsize=14)
plt.ylabel('特征', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
