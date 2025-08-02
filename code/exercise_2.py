
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
    qf=(1-0.1)*(1-x11*0.1)*(1-x12*0.1)*(1-x13*0.1)*(1-x14*0.1)*(1-x15*0.1)*(1-x16*0.1)*(1-x17*0.1)*(1-x18*0.1)*(1-x21*0.1)*(1-x22*0.1)**(1-x23*0.1)

    p=qf*200
    S1=  sum([data[f'零配件{i}']['购买单价']+x*data[f'零配件{i}']['检测成本'] for i,x in zip(range(1,9),ls1)])   
    S2=sum([data[f'半成品{i}']['装配费用']+x*data[f'半成品{i}']['检查成本'] for i,x in zip(range(1,4),ls2)]) 
    S3=data['成品']['装配费用']+x3*data['成品']['检查成本']
    A=sum([i*0.1*6 for i in [z21,z22,z23]])+z1*0.1*10
    pai_h_0=p-(x21*4+x22*4+x23*4)-(x3*6+8)-(1-qf)*z1* 10-(1-z1)*(1-qf)*30
    pai_h=(z1*(1-qf)*pai_h_0)/(1-z1*(1-qf))
    re=(1 - x3) * (1 - qf)
    E_TA=(3*(1-0.9*z21)+3*(1-0.9*z22)+2*(1-0.9*z23))/8
    pai_0=p-S1-S2-S3-A-re
    pai_c=pai_0/(E_TA)

    
    
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
x11,x12,x13,x14,x15,x16,x17,x18,x21,x22,x23,x3,z1,z21,z22,z23= best_decision
results.append({
    "最大单位利润": round(max_profit, 4),
    "零配件1检测": "是" if x11 == 1 else "否",
    "零配件2检测": "是" if x12 == 1 else "否",
    "零配件3检测": "是" if x13 == 1 else "否",
    "零配件4检测": "是" if x14 == 1 else "否",
    "零配件5检测": "是" if x15 == 1 else "否",
    "零配件6检测": "是" if x16 == 1 else "否",
    "零配件7检测": "是" if x17 == 1 else "否",
    "零配件8检测": "是" if x18 == 1 else "否",
    "半成品1检测": "是" if x21 == 1 else "否",
    "半成品1检测": "是" if x22 == 1 else "否",
    "半成品1检测": "是" if x23 == 1 else "否",
    "成品检测": "是" if x3 == 1 else "否",
     "成品拆解": "是" if z1 == 1 else "否",
     "半成品1拆解": "是" if z21 == 1 else "否",
     "半成品2拆解": "是" if z22 == 1 else "否",
        "'半成品3拆解": "是" if z23 == 1 else "否"
})

# 转换为DataFrame并显示
df = pd.DataFrame(results)
# print(df.to_string(index=False))
print(df)