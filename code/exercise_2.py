import itertools
import pandas as pd

# 表1数据（6种场景）
scenarios = [
    {"case":1, "p1":0.10,"c1":4,"d1":2, "p2":0.10,"c2":18,"d2":3, "pf":0.10,"cf":6,"df":3, "s":56,"L":6,"D":5},
    {"case":2, "p1":0.20,"c1":4,"d1":2, "p2":0.20,"c2":18,"d2":3, "pf":0.20,"cf":6,"df":3, "s":56,"L":6,"D":5},
    {"case":3, "p1":0.10,"c1":4,"d1":2, "p2":0.10,"c2":18,"d2":3, "pf":0.10,"cf":6,"df":3,"s":56 ,"L":30,"D":5},
    {"case":4, "p1":0.20,"c1":4,"d1":1, "p2":0.20,"c2":18,"d2":1, "pf":0.20,"cf":6,"df":2, "s":56,"L":30,"D":5},
    {"case":5, "p1":0.10,"c1":4,"d1":6, "p2":0.20,"c2":18,"d2":1, "pf":0.10,"cf":6,"df":2, "s":56,"L":10,"D":5},
    {"case":6, "p1":0.05,"c1":4,"d1":2, "p2":0.05,"c2":18,"d2":3, "pf":0.05,"cf":6,"df":3, "s":56,"L":10,"D":30},
]

# 计算利润的函数（修正版）
def calculate_profit(x1, x2, x3, x4, params):
    # 计算合格率
    q1 = 1 if x1 == 1 else (1 - params["p1"])  # 零件1合格概率
    q2 = 1 if x2 == 1 else (1 - params["p2"])  # 零件2合格概率
    qf = q1 * q2 * (1 - params["pf"])          # 成品合格概率
    
    # 计算各项成本和收入
    revenue = qf * params["s"]                  # 收入
    costs = (params["c1"] + params["c2"] +      # 零件成本
             x1 * params["d1"] +                # 零件1检测成本
             x2 * params["d2"] +                # 零件2检测成本
             params["cf"] +                     # 装配成本
             x3 * params["df"] +                # 成品检测成本
             params["D"] * (1 - qf) * x4 +      # 拆解成本
             (1 - x3) * (1 - qf) * params["L"]) # 售后损失
    
    # 计算单位利润
    denominator = 1 - x4 * (1 - qf)  # 考虑循环利用的影响
    if abs(denominator) < 1e-9:
        return -float('inf')
    
    return (revenue - costs) / denominator

# 遍历场景求解
results = []
for sc in scenarios:
    max_profit = -float('inf')
    best_decision = None
    
    # 枚举所有0-1组合（共16种），并过滤无效组合（x4 > x3）
    for x1, x2, x3, x4 in itertools.product([0, 1], repeat=4):
        if x4 > x3:  # 不检测成品则不能拆解
            continue
        
        current_profit = calculate_profit(x1, x2, x3, x4, sc)
        
        if current_profit > max_profit:
            max_profit = current_profit
            best_decision = (x1, x2, x3, x4)
    
    # 保存结果
    x1, x2, x3, x4 = best_decision
    results.append({
        "情况": sc["case"],
        "最大单位利润": round(max_profit, 4),
        "零配件1检测": "是" if x1 == 1 else "否",
        "零配件2检测": "是" if x2 == 1 else "否",
        "成品检测": "是" if x3 == 1 else "否",
        "不合格成品拆解": "是" if x4 == 1 else "否"
    })

# 转换为DataFrame并显示
df = pd.DataFrame(results)
# print(df.to_string(index=False))
from tabulate import tabulate

# 打印美化后的结果表格
print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
