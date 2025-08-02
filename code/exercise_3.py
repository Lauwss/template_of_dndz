
import itertools
import pandas as pd
import numpy as np

data = {"零配件1": {"次品率": 0.1,"购买单价": 2,"检测成本": 1},"零配件2": {"次品率": 0.1,"购买单价": 8, "检测成本": 1,}, "零配件3": {"次品率": 0.1, "购买单价": 12,"检测成本": 2,},"零配件4": { "次品率": 0.1,"购买单价": 2,"检测成本": 1, "半成品": {} },"零配件5": { "次品率": 0.1, "购买单价": 8,"检测成本": 1 },"零配件6": { "次品率": 0.1, "购买单价": 12,"检测成本": 2},"零配件7": { "次品率": 0.1, "购买单价": 8, "检测成本": 1, },
"零配件8" : {"次品率": 0.1,"购买单价": 12,"检测成本": 2, },"半成品1" :{"次品率": 0.1,"装配费用":8,"检查成本":4,"拆解费用":6},"半成品2" :{"次品率": 0.1,"装配费用":8,"检查成本":4,"拆解费用":6},"半成品3" :{"次品率": 0.1,"装配费用":8,"检查成本":4,"拆解费用":6},"成品":{"次品率": 0.1,"装配费用":8,"检查成本":6,"拆解费用":10,"市场售价":200,"调换损失":30}
}

# 计算利润的函数
def calculate_profit(x11,x12,x13,x14,x15,x16,x17,x18,x21,x22,x23,x3,z1,z21,z22,z23, data):
    # 计算合格率
    ls1=[x11,x12,x13,x14,x15,x16,x17,x18]
    ls2=[x21,x22,x23]
    def quality(x):  # 检测返回1，不检测返回0.9
        return 1 if x == 1 else 0.9
    qf = quality(x11)*quality(x12)*quality(x13)*quality(x14)*quality(x15)*quality(x16)*quality(x17)*quality(x18)*quality(x21)*quality(x22)*quality(x23)*0.9

    p=qf*200
    S1=  sum([data[f'零配件{i}']['购买单价']+x*data[f'零配件{i}']['检测成本'] for i,x in zip(range(1,9),ls1)])   
    S2=sum([data[f'半成品{i}']['装配费用']+x*data[f'半成品{i}']['检查成本'] for i,x in zip(range(1,4),ls2)]) 
    S3=data['成品']['装配费用']+x3*data['成品']['检查成本']
    A=sum([i*0.1*6 for i in [z21,z22,z23]])+z1*0.1*10
    pai_h_0=p-(x21*4+x22*4+x23*4)-(x3*6+8)-(1-qf)*z1* 10-(1-x3)*(1-qf)*30
    pai_h=(z1*(1-qf)*pai_h_0)/(1-z1*(1-qf))
    re=(1 - x3) * (1 - qf)*30
    E_TA=(3*z21*(1-0.9)*x21+3*z22*(1-0.9)*x22+2*z23*(1-0.9)*x23)/8
    pai_0=p-S1-S2-S3-A-re
    pai_c=pai_0/(1-E_TA)
     
    
    
    return pai_h+pai_c


# 遍历场景求解
results = []

max_profit = -float('inf')
best_decision = None

for x11,x12,x13,x14,x15,x16,x17,x18,x21,x22,x23,x3,z1,z21,z22,z23 in itertools.product([0, 1], repeat=16):
    if z21<=x21 and z22<=x22 and z23<=x23:
        current_profit = calculate_profit(x11,x12,x13,x14,x15,x16,x17,x18,x21,x22,x23,x3,z1,z21,z22,z23,data)
        if current_profit > max_profit:
            max_profit = current_profit
            best_decision = (x11,x12,x13,x14,x15,x16,x17,x18,x21,x22,x23,x3,z1,z21,z22,z23)

# 保存结果
# 解包最优决策变量
x11, x12, x13, x14, x15, x16, x17, x18, x21, x22, x23, x3, z1, z21, z22, z23 = best_decision

# 构造竖直排雷表格
vertical_results = [
    ("最大单位利润", round(max_profit, 4)),

    # 零配件检测
    ("零配件1检测", "是" if x11 else "否"),
    ("零配件2检测", "是" if x12 else "否"),
    ("零配件3检测", "是" if x13 else "否"),
    ("零配件4检测", "是" if x14 else "否"),
    ("零配件5检测", "是" if x15 else "否"),
    ("零配件6检测", "是" if x16 else "否"),
    ("零配件7检测", "是" if x17 else "否"),
    ("零配件8检测", "是" if x18 else "否"),

    # 半成品检测
    ("半成品1检测", "是" if x21 else "否"),
    ("半成品2检测", "是" if x22 else "否"),
    ("半成品3检测", "是" if x23 else "否"),

    # 成品检测
    ("成品检测", "是" if x3 else "否"),

    # 拆解决策
    ("成品拆解", "是" if z1 else "否"),
    ("半成品1拆解", "是" if z21 else "否"),
    ("半成品2拆解", "是" if z22 else "否"),
    ("半成品3拆解", "是" if z23 else "否")
]

# 计算中间项并返回组成公式各项
def calculate_profit_components(x11,x12,x13,x14,x15,x16,x17,x18,x21,x22,x23,x3,z1,z21,z22,z23, data):
    ls1 = [x11,x12,x13,x14,x15,x16,x17,x18]
    ls2 = [x21,x22,x23]
    
    def quality(x):
        return 1 if x == 1 else 0.9

    qf = quality(x11)*quality(x12)*quality(x13)*quality(x14)*quality(x15)*quality(x16)*quality(x17)*quality(x18)*quality(x21)*quality(x22)*quality(x23)*0.9
    p = qf * 200
    S1 = sum([data[f'零配件{i}']['购买单价'] + x * data[f'零配件{i}']['检测成本'] for i, x in zip(range(1, 9), ls1)])
    S2 = sum([data[f'半成品{i}']['装配费用'] + x * data[f'半成品{i}']['检查成本'] for i, x in zip(range(1, 4), ls2)])
    S3 = data['成品']['装配费用'] + x3 * data['成品']['检查成本']
    A = sum([i * 0.1 * 6 for i in [z21, z22, z23]]) + z1 * 0.1 * 10
    pai_h_0 = p - (x21 * 4 + x22 * 4 + x23 * 4) - (x3 * 6 + 8) - (1 - qf) * z1 * 10 - (1 - z1) * (1 - qf) * 30
    denominator = 1 - z1 * (1 - qf)
    pai_h = (z1 * (1 - qf) * pai_h_0 / denominator) if denominator != 0 else 0
    re = (1 - x3) * (1 - qf) * 30
    E_TA = (3 * z21 * 0.1 + 3 * z22 * 0.1 + 2 * z23 * 0.1) / 8
    denominator2 = 1 - E_TA
    pai_0 = p - S1 - S2 - S3 - A - re
    pai_c = pai_0 / denominator2 if denominator2 != 0 else 0
    total_profit = pai_h + pai_c

    return {
        "qf（合格率）": qf,
        "p（合格产品收益）": p,
        "S1（零配件成本）": S1,
        "S2（半成品成本）": S2,
        "S3（成品检测成本）": S3,
        "A（拆解成本）": A,
        "pai_h_0（补救前利润）": pai_h_0,
        "pai_h（不合格补救利润）": pai_h,
        "re（调换损失）": re,
        "E_TA（期望时间损失）": E_TA,
        "pai_0（有效利润）": pai_0,
        "pai_c（时间调整后利润）": pai_c,
        "最大单位利润（总）": total_profit
    }

# 转换为 DataFrame 并输出
df_vertical = pd.DataFrame(vertical_results, columns=["决策项", "取值"])
print(df_vertical.to_string(index=False))

# 计算详细组成项
components = calculate_profit_components(x11,x12,x13,x14,x15,x16,x17,x18,x21,x22,x23,x3,z1,z21,z22,z23, data)

# 构造中间变量结果竖直输出
print("\n📊 利润组成公式各项：\n")
df_components = pd.DataFrame(list(components.items()), columns=["计算项", "数值"])
print(df_components.to_string(index=False, float_format="%.4f"))
