import itertools
import pandas as pd
from tabulate import tabulate
import math

# 表1数据（6种场景）
scenarios = [
    {"case":1, "p1":0.10,"c1":4,"d1":2, "p2":0.10,"c2":18,"d2":3, "pf":0.10,"cf":6,"df":3, "s":56,"L":6,"D":5},
    {"case":2, "p1":0.20,"c1":4,"d1":2, "p2":0.20,"c2":18,"d2":3, "pf":0.20,"cf":6,"df":3, "s":56,"L":6,"D":5},
    {"case":3, "p1":0.10,"c1":4,"d1":2, "p2":0.10,"c2":18,"d2":3, "pf":0.10,"cf":6,"df":3,"s":56 ,"L":30,"D":5},
    {"case":4, "p1":0.20,"c1":4,"d1":1, "p2":0.20,"c2":18,"d2":1, "pf":0.20,"cf":6,"df":2, "s":56,"L":30,"D":5},
    {"case":5, "p1":0.10,"c1":4,"d1":6, "p2":0.20,"c2":18,"d2":1, "pf":0.10,"cf":6,"df":2, "s":56,"L":10,"D":5},
    {"case":6, "p1":0.05,"c1":4,"d1":2, "p2":0.05,"c2":18,"d2":3, "pf":0.05,"cf":6,"df":3, "s":56,"L":10,"D":30},
]

# 计算利润函数，返回所有关键参数列表
def calculate_profit(x1, x2, x3, x4, params, iterations=100):
    # 初始化参数
    p1 = [params["p1"]]  # 零件1初始次品率
    p2 = [params["p2"]]  # 零件2初始次品率
   
    # 初始合格率计算
    q1 = [1 if x1 == 1 else (1 - p1[0])]  # 零件1合格率
    q2 = [1 if x2 == 1 else (1 - p2[0])]  # 零件2合格率
    qf = [q1[0] * q2[0] * (1 - params["pf"])]  # 成品有效合格率
    qz = [q1[0] * q2[0] * (1 - params["pf"])]  # 理论生产合格率
    k = [((1 if x1 == 0 else (1 - p1[0])) + (1 if x2 == 0 else (1 - p2[0]))) / 2]  # 装配比例
    
    # 初始利润计算（第0次迭代）
    revenue_0 = k[0] * qf[0] * params["s"]
    costs_0 = (params["c1"] + params["c2"] +
               x1 * params["d1"] +
               x2 * params["d2"] +
               k[0] * (params["cf"] +
                       x3 * params["df"] +
                       params["D"] * (1 - qz[0]) * x4 +
                       (1 - x3) * (1 - qf[0]) * params["L"]))
    pai_0 = revenue_0 - costs_0  # 初始利润

    # 初始化迭代参数列表
    pai = []  # 利润列表
    theta_list = []  # theta值列表
    qz_list = []  # 理论合格率列表
    qf_list = []  # 有效合格率列表
    p1_list = []  # 零件1次品率列表
    p2_list = []  # 零件2次品率列表
    k_prev = k[0]
    for n in range(1, iterations + 1):
        # 更新零件次品率（若不检测则次品率上升）
        p1_n = p1[-1] / (1 - (1 - p1[-1])**2) 
        p2_n = p2[-1] / (1 - (1 - p2[-1])**2) 
        p1.append(p1_n)
        p2.append(p2_n)
        p1_list.append(p1_n)
        p2_list.append(p2_n)

        # 更新合格率
        q1_n = 1 if x1 == 1 else (1 - p1_n)  # 零件1合格率
        q2_n = 1 if x2 == 1 else (1 - p2_n)  # 零件2合格率
        qf_n = q1_n * q2_n * (1 - params["pf"])  # 有效合格率（受检测影响）
        qz_n = q1_n * q2_n * (1 - params["pf"])  # 理论合格率（不受检测影响）
        k_n = ((1 if x1 == 0 else (1 - p1_n)) + (1 if x2 == 0 else (1 - p2_n))) / 2  # 装配比例
         # 计算theta值（与拆解决策相关）
     
        # 保存当前迭代的合格率
        qz_list.append(qz_n)
        
        
        # 累积theta乘积
        theta_n = x4 * (1 - qf[-1]) * k_prev
        theta_list.append(theta_n)
        k_prev = k_n
        qf_list.append(qf_n)
        # 计算theta乘积的积
        multiply_theta = math.prod(theta_list) 
        # 计算当前迭代的利润
        revenue_n = qf_n * k_n * params["s"]
        cost_n = (x1 * params["d1"] +
                  x2 * params["d2"] +
                  k_n * (params["cf"] +
                         x3 * params["df"] +
                         params["D"] * (1 - qz_n) * x4 +
                         (1 - x3) * (1 - qf_n) * params["L"]))
        pai_n = multiply_theta * (revenue_n - cost_n)
        pai.append(pai_n)

    total_profit = sum(pai) + pai_0  # 总利润（包括初始利润）
    # 返回所有计算结果
    return (total_profit, theta_list, pai, 
            qz_list, qf_list, p1_list, p2_list,
            pai_0, qf[0], qz[0])

# 汇总结果存储
summary_results = []
detailed_records = []
best_details = []  # 存储最优策略的详细信息

for sc in scenarios:
    case_id = sc["case"]
    max_profit = -float('inf')
    best_decision = None
    # 初始化最优策略的详细参数
    best_theta = None
    best_pai = None
    best_qz = None
    best_qf = None
    best_p1 = None
    best_p2 = None
    best_pai0 = None
    best_qf0 = None
    best_qz0 = None

    # 遍历所有可能的决策组合（x1, x2, x3, x4均为0或1）
    for x1, x2, x3, x4 in itertools.product([0, 1], repeat=4):
        (profit, theta_list, pai_list, 
         qz_list, qf_list, p1_list, p2_list,
         pai_0, qf_0, qz_0) = calculate_profit(x1, x2, x3, x4, sc)
        
        # 记录详细策略结果
        detailed_records.append({
            "情况": case_id,
            "x1_零件1检测": x1,
            "x2_零件2检测": x2,
            "x3_成品检测": x3,
            "x4_拆解": x4,
            "单位利润": round(profit, 4)
        })

        # 更新最大利润组合
        if profit > max_profit:
            max_profit = profit
            best_decision = (x1, x2, x3, x4)
            best_theta = theta_list
            best_pai = pai_list
            best_qz = qz_list
            best_qf = qf_list
            best_p1 = p1_list
            best_p2 = p2_list
            best_pai0 = pai_0
            best_qf0 = qf_0
            best_qz0 = qz_0

    # 保存最优策略到汇总结果
    x1, x2, x3, x4 = best_decision
    summary_results.append({
        "情况": case_id,
        "最大单位利润": round(max_profit, 4),
        "零配件1检测": "是" if x1 == 1 else "否",
        "零配件2检测": "是" if x2 == 1 else "否",
        "成品检测": "是" if x3 == 1 else "否",
        "不合格成品拆解": "是" if x4 == 1 else "否"
    })
    
    # 保存最优策略的详细信息
    best_details.append({
        "case": case_id,
        "decision": best_decision,
        "total_profit": max_profit,
        "theta_list": best_theta,
        "pai_list": best_pai,
        "qz_list": best_qz,
        "qf_list": best_qf,  # 有效合格率列表
        "p1_list": best_p1,
        "p2_list": best_p2,
        "pai_0": best_pai0,
        "qf_0": best_qf0,  # 初始有效合格率
        "qz_0": best_qz0
    })

# 输出最优策略汇总表
df_summary = pd.DataFrame(summary_results)
print("\n【最优策略汇总】")
print(tabulate(df_summary, headers="keys", tablefmt="grid", showindex=False))

# 输出每种情形最优策略的qf（有效合格率）信息
for detail in best_details:
    case_id = detail["case"]
    decision = detail["decision"]
    qf_0 = detail["qf_0"]  # 初始有效合格率
    qf_list = detail["qf_list"]  # 迭代过程中的有效合格率
    
    print(f"\n\n【情况 {case_id} 最优策略的有效合格率(qf)】")
    print(f"最优决策: x1={decision[0]}（零件1检测：{'是' if decision[0]==1 else '否'}）, "
          f"x2={decision[1]}（零件2检测：{'是' if decision[1]==1 else '否'}）, "
          f"x3={decision[2]}（成品检测：{'是' if decision[2]==1 else '否'}）, "
          f"x4={decision[3]}（拆解：{'是' if decision[3]==1 else '否'}）")
    
  
