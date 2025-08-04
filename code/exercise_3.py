import itertools
import pandas as pd
import numpy as np
import math
from tqdm import tqdm 
from deap import base, creator, tools, algorithms
import random

# 参数数据
data = {
    "零配件1": {"次品率": 0.1, "购买单价": 2, "检测成本": 1},
    "零配件2": {"次品率": 0.1, "购买单价": 8, "检测成本": 1},
    "零配件3": {"次品率": 0.1, "购买单价": 12, "检测成本": 2},
    "零配件4": {"次品率": 0.1, "购买单价": 2, "检测成本": 1},
    "零配件5": {"次品率": 0.1, "购买单价": 8, "检测成本": 1},
    "零配件6": {"次品率": 0.1, "购买单价": 12, "检测成本": 2},
    "零配件7": {"次品率": 0.1, "购买单价": 8, "检测成本": 1},
    "零配件8": {"次品率": 0.1, "购买单价": 12, "检测成本": 2},
    "半成品1": {"次品率": 0.1, "装配费用": 8, "检查成本": 4, "拆解费用": 6},
    "半成品2": {"次品率": 0.1, "装配费用": 8, "检查成本": 4, "拆解费用": 6},
    "半成品3": {"次品率": 0.1, "装配费用": 8, "检查成本": 4, "拆解费用": 6},
    "成品": {"次品率": 0.1, "装配费用": 8, "检查成本": 6, "拆解费用": 10, "市场售价": 200, "调换损失": 30}
}


def calculate_profit(x, data, iterations=1000):
    x11,x12,x13,x14,x15,x16,x17,x18,x21,x22,x23,x3,z1,z21,z22,z23 = x
    ls1 = [x11,x12,x13,x14,x15,x16,x17,x18]
    ls2 = [x21,x22,x23]
    
    # 基础参数初始化
    p11 =[data["零配件1"]["次品率"]]
    p12 =[data["零配件2"]["次品率"]]
    p13 =[data["零配件3"]["次品率"]]
    p14 =[data["零配件4"]["次品率"]]
    p15 =[data["零配件5"]["次品率"]]
    p16 =[data["零配件6"]["次品率"]]
    p17 =[data["零配件7"]["次品率"]]
    p18 =[data["零配件8"]["次品率"]]
    p21 =[data["半成品1"]["次品率"]]
    p22 =[data["半成品2"]["次品率"]]
    p23 =[data["半成品3"]["次品率"]]
    components_p11 =[data["零配件1"]["次品率"]]
    components_p12 =[data["零配件2"]["次品率"]]
    components_p13 =[data["零配件3"]["次品率"]]
    components_p14 =[data["零配件4"]["次品率"]]
    components_p15 =[data["零配件5"]["次品率"]]
    components_p16 =[data["零配件6"]["次品率"]]
    components_p17 =[data["零配件7"]["次品率"]]
    components_p18 =[data["零配件8"]["次品率"]]
    pf = data["成品"]["次品率"]
    
    # 合格率计算
    q11=[1 - p11[0] if ls1[0]==0 else 1]
    q12=[1 - p12[0] if ls1[1]==0 else 1]
    q13=[1 - p13[0] if ls1[2]==0 else 1]
    q14=[1 - p14[0] if ls1[3]==0 else 1]
    q15=[1 - p15[0] if ls1[4]==0 else 1]
    q16=[1 - p16[0] if ls1[5]==0 else 1]
    q17=[1 - p17[0] if ls1[6]==0 else 1]
    q18=[1 - p18[0] if ls1[7]==0 else 1]
    q21 = [q11[0] * q12[0] * q13[0] * (1 - p21[0] if ls2[0]==0 else 1)]
    q22 = [q14[0] * q15[0] * q16[0] * (1 - p22[0] if ls2[1]==0 else 1)]
    q23 = [q17[0] * q18[0] * (1 - p23[0] if ls2[2]==0 else 1)]
    qf_initial = q21[0] * q22[0] * q23[0] * (1 - pf)
    qf = [qf_initial]  # 存储所有迭代的qf值，初始值为第0次
    
    # 装配比例计算
    k11=[((1 - p11[0] if ls1[0]==1 else 1)+(1 - p12[0] if ls1[1]==1 else 1)+(1 - p13[0] if ls1[2]==1 else 1))/3]
    k12=[((1 - p14[0] if ls1[3]==1 else 1)+(1 - p15[0] if ls1[4]==1 else 1)+(1 - p16[0] if ls1[5]==1 else 1))/3]
    k13=[((1 - p17[0] if ls1[6]==1 else 1)+(1 - p18[0] if ls1[7]==1 else 1))/2]
    k2=[(k11[0]*q11[0] * q12[0] * q13[0] * (1 - p21[0] if ls2[0]==1 else 1)+
         k12[0]*q14[0] * q15[0] * q16[0] * (1 - p22[0] if ls2[1]==1 else 1)+
         k13[0]*q17[0] * q18[0] * (1 - p23[0] if ls2[2]==1 else 1))/3]
    
    # 初始利润构成（第0次迭代）
    revenue_0 = qf[0] * data["成品"]["市场售价"]  # 初始收入
    S1 = sum([data[f'零配件{i}']['购买单价'] + x * data[f'零配件{i}']['检测成本'] 
             for i, x in zip(range(1, 9), ls1)])  # 零配件总成本
    S2 = sum([k*data[f'半成品{i}']['装配费用'] + k*x * data[f'半成品{i}']['检查成本'] 
             for i, x, k in zip(range(1, 4), ls2,[k11[0],k12[0],k13[0]])])  # 半成品总成本
    S3 = k2[0]*(data['成品']['装配费用'] + x3 * data['成品']['检查成本'])  # 成品总成本
    A = (k2[0] * z1 * (1-qf[0]) * data['成品']['拆解费用'] + 
         sum([z * (1 - q2) * data[f'半成品{idx+1}']['拆解费用'] 
             for idx, (z, q2) in enumerate(zip([z21, z22, z23], [q21[0], q22[0], q23[0]]))]))  # 拆解总成本
    re = k2[0]*(1 - x3) * (1 - qf[0]) * data['成品']['调换损失']  # 调换损失成本
    cost_0 = S1 + S2 + S3 + A + re  # 初始总成本
    pai_0 = revenue_0 - cost_0  # 初始利润
    
    # 迭代利润累积
    half_pai = []  # 半成品相关利润（保留迭代过程）
    components_pai = []  # 零配件相关利润
    theta_list = []  
    beta_list = []  
    first_iteration_details = None  # 存储第一次迭代的详细信息
    theta_first_breakdown = None  # 存储theta第一项的构成
    k2_prev = k2[0]  # 存储上一次迭代的k2值
    
    for n in range(1, iterations + 1):
        # 更新次品率与合格率
        p21_n = p21[-1]  / (1 - (1 - p21[-1])**3)
        p22_n = p22[-1] / (1 - (1 - p22[-1])**3)
        p23_n = p23[-1] / (1 - (1 - p23[-1])**3)
        p21.append(p21_n)
        p22.append(p22_n)
        p23.append(p23_n)
        
        q21_n = q11[0]*q12[0]*q13[0] *(1 - p21_n if ls2[0]==0 else 1)
        q22_n = q14[0]*q15[0]*q16[0] *(1 - p22_n if ls2[1]==0 else 1)
        q23_n = q17[0]*q18[0] *(1 - p23_n if ls2[2]==0 else 1)
        qf_n = q21_n * q22_n * q23_n * (1 - pf)
      
      
        k2_n = (q11[0]*q12[0]*q13[0] *(1 - p21_n if ls2[0]==1 else 1) + 
                q14[0]*q15[0]*q16[0]*(1 - p22_n if ls2[1]==0 else 1) + 
                q17[0]*q18[0]*(1 - p23_n if ls2[2]==0 else 1)) / 3
       
        # 更新零配件次品率
        components_p11_n = components_p11[-1]  / (1 - (1 - components_p11[-1])**3)
        components_p12_n = components_p12[-1]  / (1 - (1 - components_p12[-1])**3)
        components_p13_n = components_p13[-1]  / (1 - (1 - components_p13[-1])**3)
        components_p14_n = components_p14[-1]  / (1 - (1 - components_p14[-1])**3)
        components_p15_n = components_p15[-1]  / (1 - (1 - components_p15[-1])**3)
        components_p16_n = components_p16[-1]  / (1 - (1 - components_p16[-1])**3)
        components_p17_n = components_p17[-1] / (1 - (1 - components_p17[-1])**2)
        components_p18_n = components_p18[-1] / (1 - (1 - components_p18[-1])**2)
        components_p11.append(components_p11_n)
        components_p12.append(components_p12_n)
        components_p13.append(components_p13_n)
        components_p14.append(components_p14_n)
        components_p15.append(components_p15_n)
        components_p16.append(components_p16_n)
        components_p17.append(components_p17_n)
        components_p18.append(components_p18_n)

        # 计算组件合格率
        components_q11_n = 1 - components_p11_n if ls1[0]==0 else 1
        components_q12_n = 1 - components_p12_n if ls1[1]==0 else 1
        components_q13_n = 1 - components_p13_n if ls1[2]==0 else 1
        components_q14_n = 1 - components_p14_n if ls1[3]==0 else 1
        components_q15_n = 1 - components_p15_n if ls1[4]==0 else 1
        components_q16_n = 1 - components_p16_n if ls1[5]==0 else 1
        components_q17_n = 1 - components_p17_n if ls1[6]==0 else 1
        components_q18_n = 1 - components_p18_n if ls1[7]==0 else 1
        
        # 计算半成品组件合格率
        components_q21_n = components_q11_n * components_q12_n * components_q13_n * (1 - p21_n if ls2[0]==1 else 1)
        components_q22_n = components_q14_n * components_q15_n * components_q16_n * (1 - p22_n if ls2[1]==1 else 1)
        components_q23_n = components_q17_n * components_q18_n * (1 - p23_n if ls2[2]==1 else 1)

        # 计算迭代中的利润组件
        components_qf_n = components_q21_n * components_q22_n * components_q23_n * (1 - pf )
        
        components_k2_n = (
            ((1 - p11[-1] if ls1[0]==1 else 1)+(1 - p12[-1] if ls1[1]==1 else 1)+(1 - p13[-1] if ls1[2]==1 else 1))/3 *
            components_q11_n * components_q12_n * components_q13_n * (1 - p21_n if ls2[0]==1 else 1) +
            ((1 - p14[-1] if ls1[3]==1 else 1)+(1 - p15[-1] if ls1[4]==1 else 1)+(1 - p16[-1] if ls1[5]==1 else 1))/3 *
            components_q14_n * components_q15_n * components_q16_n * (1 - p22_n if ls2[1]==1 else 1) +
            ((1 - p17[-1] if ls1[6]==1 else 1)+(1 - p18[-1] if ls1[7]==1 else 1))/2 *
            components_q17_n * components_q18_n * (1 - p23_n if ls2[2]==1 else 1)
        ) / 3

        # 累积参数与利润
        beta_n = (3*z21*(1 - q21_n) + 3*z22*(1 - q22_n) + 2*z23*(1 - q23_n))/8
        theta_n = z1 * (1 - qf[-1])*k2_prev + z1*(1 - components_qf_n)*components_k2_n * beta_n
        qf.append(qf_n)  # 保存当前迭代的qf值
        
        k2_prev = k2_n
        theta_list.append(theta_n)
        beta_list.append(beta_n)
        
        multiply_theta = math.prod(theta_list)
        multiply_beta = math.prod(beta_list)
        
        # 半成品相关利润（每次迭代结果存入half_pai）
        half_revenue_n = k2_n * qf_n * data["成品"]["市场售价"]
        half_cost_n = (sum([x * data[f'半成品{i}']['检查成本'] for i, x in zip(range(1, 4), ls2)]) +
                      z21*(1 - q21_n)*data['半成品1']['拆解费用'] + 
                      z22*(1 - q22_n)*data['半成品2']['拆解费用'] + 
                      z23*(1 - q23_n)*data['半成品3']['拆解费用'] +  
                      k2_n*(data['成品']['装配费用'] + x3 * data['成品']['检查成本'] +
                            (1 - qf_n) * z1 * data['成品']['拆解费用'] +
                            (1 - x3) * (1 - qf_n) * data['成品']['调换损失']))
        current_half_pai = multiply_theta * (half_revenue_n - half_cost_n)
        half_pai.append(current_half_pai)
        
        # 零配件相关利润
        components_revnue_n = components_k2_n * components_qf_n * data["成品"]["市场售价"]
        components_S1_n = sum([x * data[f'零配件{i}']['检测成本'] for i, x in zip(range(1, 9), ls1)])
        components_S2_n = sum([k * data[f'半成品{i}']['装配费用'] + k * x * data[f'半成品{i}']['检查成本'] 
                              for i, x, k in zip(range(1, 4), ls2, [k11[0], k12[0], k13[0]])])
        components_S3_n = components_k2_n * (data['成品']['装配费用'] + x3 * data['成品']['检查成本'])
        components_A_n = (components_k2_n * z1 * (1 - components_qf_n) * data['成品']['拆解费用'] + 
                         sum([z * (1 - components_q21_n if idx==0 else components_q22_n if idx==1 else components_q23_n) * 
                              data[f'半成品{idx+1}']['拆解费用'] 
                              for idx, z in enumerate([z21, z22, z23])]))
        components_re = components_k2_n * (1 - x3) * (1 - components_qf_n) * 30
        components_cost_n = components_S1_n + components_S2_n + components_S3_n + components_A_n + components_re
        components_pai.append(multiply_theta * multiply_beta * (components_revnue_n - components_cost_n))
    
    # 总利润及构成汇总
    total_iteration_profit = sum(half_pai) + sum(components_pai)  # 迭代累积利润
    total_profit = total_iteration_profit + pai_0  # 总利润
    
    # 返回利润构成及相关详细信息
    return {
        "总利润": total_profit,
    }


def calculate_qf(ls1, ls2, data):
    """计算成品合格率"""
    q11 = 1 - data["零配件1"]["次品率"] if ls1[0] == 0 else 1
    q12 = 1 - data["零配件2"]["次品率"] if ls1[1] == 0 else 1
    q13 = 1 - data["零配件3"]["次品率"] if ls1[2] == 0 else 1
    q14 = 1 - data["零配件4"]["次品率"] if ls1[3] == 0 else 1
    q15 = 1 - data["零配件5"]["次品率"] if ls1[4] == 0 else 1
    q16 = 1 - data["零配件6"]["次品率"] if ls1[5] == 0 else 1
    q17 = 1 - data["零配件7"]["次品率"] if ls1[6] == 0 else 1
    q18 = 1 - data["零配件8"]["次品率"] if ls1[7] == 0 else 1
    q21 = q11 * q12 * q13 * (1 - data["半成品1"]["次品率"] if ls2[0] == 0 else 1)
    q22 = q14 * q15 * q16 * (1 - data["半成品2"]["次品率"] if ls2[1] == 0 else 1)
    q23 = q17 * q18 * (1 - data["半成品3"]["次品率"] if ls2[2] == 0 else 1)
    return q21 * q22 * q23 * (1 - data["成品"]["次品率"])


# 设置遗传算法参数
POP_SIZE = 100     # 种群大小
N_GEN = 10         # 迭代代数
CX_PB = 0.5        # 交叉概率
MUT_PB = 0.2       # 变异概率

# 创建适应度和个体类
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # 最大化问题
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=16)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 评估函数（带约束判断）
def eval_profit(ind):
    x11,x12,x13,x14,x15,x16,x17,x18,x21,x22,x23,x3,z1,z21,z22,z23 = ind
    if z21 > x21 or z22 > x22 or z23 > x23:
        return -1e6,  # 非可行解惩罚
    result = calculate_profit(ind, data)
    return result["总利润"],

toolbox.register("evaluate", eval_profit)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# 初始化种群
population = toolbox.population(n=POP_SIZE)

# 进化过程
print("开始遗传算法优化...")
for gen in tqdm(range(N_GEN), desc="进化中", ncols=80):
    offspring = algorithms.varAnd(population, toolbox, cxpb=CX_PB, mutpb=MUT_PB)
    fits = list(map(toolbox.evaluate, offspring))
    for ind, fit in zip(offspring, fits):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
print("优化完成。")

# 选出最优个体
best_ind = tools.selBest(population, k=1)[0]
best_profit_result = calculate_profit(best_ind, data)

max_profit = best_profit_result["总利润"]
best_decision = best_ind
profit_components = best_profit_result

# 输出最佳决策
labels = ["零配件1","零配件2","零配件3","零配件4","零配件5","零配件6","零配件7","零配件8",
          "半成品1","半成品2","半成品3","成品","成品拆解","半成品1拆解","半成品2拆解","半成品3拆解"]
choices = ["是" if i else "否" for i in best_decision]
results_df = pd.DataFrame(zip(["最大单位利润（总）"] + labels, [round(max_profit, 4)] + choices), 
                         columns=["决策项", "取值"])
print("\n最佳决策方案：")
print(results_df.to_string(index=False))



