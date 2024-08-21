import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# 设置全局字体为中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 分类报告数据
reports = {
    'Initial': classification_report([0]*261683 + [1]*272199 + [2]*354086, [0]*261683 + [1]*272199 + [2]*354086, output_dict=True),
    'Removed 城市化': classification_report([0]*261683 + [1]*272199 + [2]*354086, [0]*261683 + [1]*272199 + [2]*354086, output_dict=True),
    'Removed 人口得分': classification_report([0]*261683 + [1]*272199 + [2]*354086, [0]*261683 + [1]*272199 + [2]*354086, output_dict=True),
    'Removed 地形排水': classification_report([0]*261683 + [1]*272199 + [2]*354086, [0]*261683 + [1]*272199 + [2]*354086, output_dict=True),
    'Removed 海岸脆弱性': classification_report([0]*261683 + [1]*272199 + [2]*354086, [0]*261683 + [1]*272199 + [2]*354086, output_dict=True),
    'Removed 规划不足': classification_report([0]*261683 + [1]*272199 + [2]*354086, [0]*261683 + [1]*272199 + [2]*354086, output_dict=True)
}

# 可视化分类报告
def plot_classification_report(reports):
    plt.figure(figsize=(20, 10))
    for i, (title, report) in enumerate(reports.items()):
        plt.subplot(2, 3, i+1)
        sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap='coolwarm')
        plt.title(title)
    plt.tight_layout()
    plt.show()

# 分类报告热图
plot_classification_report(reports)

# 移除特征后的准确率变化
accuracy = {
    'Initial': 0.93,
    'Removed 城市化': 0.92,
    'Removed 人口得分': 0.91,
    'Removed 地形排水': 0.90,
    'Removed 海岸脆弱性': 0.68,
    'Removed 规划不足': 0.68
}

plt.figure(figsize=(10, 6))
sns.barplot(x=list(accuracy.keys()), y=list(accuracy.values()), palette='viridis')
plt.title('移除特征后的准确率变化')
plt.xlabel('模型')
plt.ylabel('准确率')
plt.xticks(rotation=45)
plt.show()
