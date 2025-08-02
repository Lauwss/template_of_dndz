import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 固定参数
n = 865
alpha = 0.05
Z_alpha = norm.ppf(1 - alpha)
p0 = 0.10

# 计算临界比例
p_crit = p0 + Z_alpha * np.sqrt(p0 * (1 - p0) / n)

# 构造不同实际不合格率
p_values = np.linspace(0.01, 0.20, 100)
accept_probs = []

for p in p_values:
    std = np.sqrt(p * (1 - p) / n)
    prob = norm.cdf((p_crit - p) / std)
    accept_probs.append(prob)

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(p_values, accept_probs, label='OC Curve (n=865, α=0.05)')
plt.axvline(p0, color='gray', linestyle='--', label=f'p0 = {p0}')
plt.axhline(0.95, color='r', linestyle=':', label='95% Acceptance Level')
plt.xlabel('Actual Defective Rate $p$')
plt.ylabel('Probability of Acceptance')
plt.title('Operating Characteristic (OC) Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(r'C:\Users\20342\Downloads\template_of_dndz\figure\OC_curve_1.png')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 固定参数
n = 609
alpha = 0.1
Z_alpha = norm.ppf(1 - alpha)
p0 = 0.10

# 计算临界比例
p_crit = p0 + Z_alpha * np.sqrt(p0 * (1 - p0) / n)
print(p_crit)
# 构造不同实际不合格率
p_values = np.linspace(0.01, 0.20, 100)
accept_probs = []

for p in p_values:
    std = np.sqrt(p * (1 - p) / n)
    prob = norm.cdf((p_crit - p) / std)
    accept_probs.append(prob)

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(p_values, accept_probs, label='OC Curve (n=609, α=0.1)')
plt.axvline(p0, color='gray', linestyle='--', label=f'p0 = {p0}')
plt.axhline(0.90, color='r', linestyle=':', label='90% Acceptance Level')
plt.xlabel('Actual Defective Rate $p$')
plt.ylabel('Probability of Acceptance')
plt.title('Operating Characteristic (OC) Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(r'C:\Users\20342\Downloads\template_of_dndz\figure\OC_curve_2.png')

