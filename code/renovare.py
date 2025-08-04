import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文黑体
plt.rcParams['axes.unicode_minus'] = False

cases = [1, 2, 3, 4, 5, 6]
profits = [13.4047, 4.175, 11.139, 6.9998, 11.2487, 18.5867]

strategy = {
    "零件1检测": [1,1,1,1,0,0],
    "零件2检测": [1,1,1,1,1,0],
    "成品检测": [0,0,0,1,1,0],
    "拆解": [1,1,1,1,1,0]
}

cost_data = {
    1: {"零件1检测成本": 2, "零件2检测成本": 3, "成品检测成本": 3, "拆解费用": 5},
    2: {"零件1检测成本": 2, "零件2检测成本": 3, "成品检测成本": 3, "拆解费用": 5},
    3: {"零件1检测成本": 2, "零件2检测成本": 3, "成品检测成本": 3, "拆解费用": 5},
    4: {"零件1检测成本": 1, "零件2检测成本": 1, "成品检测成本": 2, "拆解费用": 5},
    5: {"零件1检测成本": 6, "零件2检测成本": 1, "成品检测成本": 2, "拆解费用": 5},
    6: {"零件1检测成本": 2, "零件2检测成本": 3, "成品检测成本": 3, "拆解费用": 30},
}

fixed_costs = {
    1: {"零件成本": 4+18, "成品装配成本": 6},
    2: {"零件成本": 4+18, "成品装配成本": 6},
    3: {"零件成本": 4+18, "成品装配成本": 6},
    4: {"零件成本": 4+18, "成品装配成本": 6},
    5: {"零件成本": 4+18, "成品装配成本": 6},
    6: {"零件成本": 4+18, "成品装配成本": 6},
}

costs_list = []
for c in cases:
    cd = cost_data[c]
    fc = fixed_costs[c]
    strat = {k: strategy[k][c-1] for k in strategy}

    零件检测成本 = strat["零件1检测"] * cd["零件1检测成本"] + strat["零件2检测"] * cd["零件2检测成本"]
    成品检测成本 = strat["成品检测"] * cd["成品检测成本"]
    拆解成本 = strat["拆解"] * cd["拆解费用"]
    零件成本 = fc["零件成本"]
    成品成本 = fc["成品装配成本"]

    costs_list.append({
        "零件检测成本": 零件检测成本,
        "成品检测成本": 成品检测成本,
        "拆解成本": 拆解成本,
        "零件成本": 零件成本,
        "成品装配成本": 成品成本,
    })

df_costs = pd.DataFrame(costs_list, index=cases)

x = np.arange(len(cases))
bar_width = 0.15
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [1, 2]})

# 上面策略条形图
for i, (key, values) in enumerate(strategy.items()):
    ax1.bar(x + i * bar_width, values, bar_width, label=key)

ax1.set_xticks(x + bar_width * 1.5)
ax1.set_xticklabels([f"情况 {c}" for c in cases])
ax1.set_ylim(0, 1.2)
ax1.set_yticks([0, 1])
ax1.set_yticklabels(["否", "是"])
ax1.set_title("各情况最优策略选择")
ax1.legend(loc='upper right')

# 下面利润和成本堆积图
bottom = np.zeros(len(cases))
cost_keys = ["零件成本", "零件检测成本", "成品检测成本", "拆解成本", "成品装配成本"]
for i, key in enumerate(cost_keys):
    ax2.bar(x, df_costs[key], bottom=bottom, label=key, color=colors[i])
    bottom += df_costs[key]

ax2.plot(x, profits, color='black', marker='o', linewidth=2, label='最大单位利润')

ax2.set_xticks(x)
ax2.set_xticklabels([f"情况 {c}" for c in cases])
ax2.set_ylabel("金额（元）")
ax2.set_title("各情况利润与成本构成")
ax2.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.savefig(r'C:\Users\20342\Downloads\template_of_dndz\figure\fuck.png')
plt.tight_layout()
plt.show()
