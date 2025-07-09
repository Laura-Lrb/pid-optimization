import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

if __name__ == "__main__":
    # 加载数据（替换为实际文件路径）
    file_path = "temperature_data.csv"
    df = pd.read_csv(file_path)
    t = df['time'].values  # 时间列（单位：秒）
    y = df['temperature'].values  # 温度列（单位：℃）
    u = df['volte'].values  # 电压列（单位：V）
    percent = (y-y[0])/(y[-1]-y[0])
    percent = savgol_filter(percent, window_length=11, polyorder=3)
    # 查找最接近 0.393 和 0.632 的点的索引
    target1 = 0.284
    target2= 0.632
    # t1 = np.argmin(np.abs(percent - target1))  # 找到最接近 0.393 的点
    # t2 = np.argmin(np.abs(percent - target2))  # 找到最接近 0.632 的点
    f = interp1d(percent, t, kind='linear', fill_value="extrapolate")
    t1 = f(target1)
    t2 = f(target2)
    # 输出结果
    print(f"通过插值得到纵坐标为 {target1:.2f} 的时间点为: t = {t1:.2f}s")
    print(f"通过插值得到纵坐标为 {target2:.2f} 的时间点为: t = {t2:.2f}s")
    T = (t2 - t1) / (np.log(1 - target1) - np.log(1 - target2))
    tao=(t2*np.log(1 - target1)-t1*np.log(1 - target2)) / (np.log(1 - target1) - np.log(1 - target2))
    K = (y[-1] - y[0])/u[0]
    print("T:",T)
    print("t:",tao)
    print("k:", K)
    # print("t:",t[t1]-0.33*T)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
    # 绘图并标注这两个点
    plt.figure(figsize=(12, 6))
    plt.plot(t, percent, 'b.-', label='归一化温度', markersize=4)
    # plt.plot(t[t1], percent[t1], 'ro', label=f'0.393 点 (t={t[t1]:.2f})')
    # plt.plot(t[t2], percent[t2], 'go', label=f'0.632 点 (t={t[t2]:.2f})')

    # 添加水平参考线
    plt.axhline(y=target1, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=target2, color='g', linestyle='--', alpha=0.5)

    plt.xlabel('时间 (s)')
    plt.ylabel('归一化温度')
    plt.title('查找 0.393 和 0.632 的特征点')
    plt.legend()
    plt.grid(True)
    plt.show()

