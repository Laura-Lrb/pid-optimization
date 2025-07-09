import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import control

# 系统参数
K = 9.851946150625     # 增益
tau = 2998.4622211064966  # 时间常数
delay = 105.16771051671549              # 延迟时间
order = 3                # Pade 近似阶数（推荐 1~3）

# 创建一阶系统 G(s) = K / (tau*s + 1)
sys_1st_order = control.TransferFunction([K], [tau, 1])

# 使用 Pade 近似创建延迟环节
num_delay, den_delay = control.pade(delay, order)
sys_delay = control.TransferFunction(num_delay, den_delay)

# 将延迟环节与原系统串联
sys_with_delay = control.series(sys_delay, sys_1st_order)

# 输出整个系统的传递函数
print("带延迟的一阶系统：")
print(sys_with_delay)

# 生成仿真数据
t = np.arange(0, 10800.5, 0.5)
u = np.ones_like(t)*3.5
t_sim, y_sim = control.forced_response(sys_with_delay, T=t, U=u)
t_cal = np.arange(0, 10800.5, 0.5)

# 加载实验数据（假设CSV格式）
data = pd.read_csv('temperature_data.csv')
t_exp = data['time'].values
y_exp = data['temperature'].values
y_exp = y_exp - y_exp[0]

# 计算残差（对齐长度）
residuals = y_exp - y_sim[:len(y_exp)]

# 可视化
plt.figure(figsize=(12, 8))

# 时域对比
plt.subplot(2, 2, 1)
plt.plot(t_exp, y_exp, 'b-', label='Experimental')
plt.plot(t_sim, y_sim, 'r--', label='Model with Delay')
# plt.plot(t_cal,y_cal,'g-',label='calculation')
plt.legend()
plt.grid(True)

# 残差散点图
plt.subplot(2, 2, 2)
plt.scatter(y_sim[:len(y_exp)], residuals)
plt.grid(True)

# 残差分布
plt.subplot(2, 2, 3)
plt.hist(residuals, bins=20)

# ACF图
from statsmodels.graphics.tsaplots import plot_acf
plt.subplot(2, 2, 4)
plot_acf(residuals, lags=50)

plt.tight_layout()
plt.show()