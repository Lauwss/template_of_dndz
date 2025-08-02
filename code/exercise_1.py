import math
from scipy.stats import norm

# 定义参数
p0 = 0.10  # 标称次品率
alpha_95 = 0.05  # 95%置信水平
alpha_90 = 0.10  # 90%置信水平
z_95 = norm.ppf(1-alpha_95/2) # 95%的临界值
z_90 = norm.ppf(1-alpha_90/2 )  # 90%的临界值

# 计算样本量
def calculate_sample_size(z_alpha, p0, delta):
    n=(z_alpha/delta)**2*p0*(1-p0)
    return math.ceil(n)
# 假设检测误差 delta 为 5%
delta = 0.02
n_95=calculate_sample_size(z_95,p0,delta)
n_90=calculate_sample_size(z_90,p0,delta)
print(n_95,n_90)