import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文黑体
plt.rcParams['axes.unicode_minus'] = False
# 决策数据
strategy = {
    '零配件1': 1, '零配件2': 1, '零配件3': 1, '零配件4': 1,
    '零配件5': 1, '零配件6': 1, '零配件7': 1, '零配件8': 1,
    '半成品1': 1, '半成品2': 1, '半成品3': 1,
    '成品': 0,
    '半成品1拆解': 0, '半成品2拆解': 1, '半成品3拆解': 1,
    '成品拆解': 1
}
df = pd.DataFrame(strategy.items(), columns=["项目", "是否选择"])
heatmap_data = df.pivot_table(index='项目', values='是否选择', aggfunc='sum').sort_index(ascending=False)

# 收益构成
labels = ['总收益（销售收入）', '零件购买', '检测成本', '装配成本', '拆解费用', '售后损失', '利润']
values = [200, 66, 20, 24, 16, 0, 74.83]
colors = ["green", "red", "orange", "orange", "orange", "red", "blue"]

# 绘图
fig, axes = plt.subplots(2, 1, figsize=(12, 10))  # 两行一列的子图布局

# 子图1：策略热力图
sns.heatmap(heatmap_data, cmap='YlGnBu', cbar=False, annot=True, ax=axes[0])
axes[0].set_title("最佳策略下各检测/拆解项目选择图")
axes[0].set_xticks([])

# 子图2：单位利润构成柱状图
bars = axes[1].bar(labels, values, color=colors)
axes[1].set_ylabel("金额 (元)")
axes[1].set_title("单位利润构成分析图")
axes[1].tick_params(axis='x', rotation=45)

# 为每个柱子添加数值标签
for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}', ha='center', va='bottom')
plt.savefig(r'C:\Users\20342\Downloads\template_of_dndz\figure\final.png')
plt.tight_layout()
plt.show()
