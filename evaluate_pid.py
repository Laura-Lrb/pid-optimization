import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd

# 干扰信号(正弦波)
def disturbance(t):
    # return 0.5 * np.sin(2*np.pi*t/3600)  # 周期1小时的扰动
    return 0
# 含延迟的系统模型
def system(y, t, u_history, t_history):
    # 系统参数
    K = 9.851946150625  # 增益
    tau = 2998.4622211064966  # 时间常数
    delay = 105.16771051671549  # 延迟时间
    # 当前时间的延迟控制量(线性插值)
    # delay = 0
    u_delayed = np.interp(t - delay, t_history, u_history, left=0)
    dydt = (K * u_delayed - y) / tau
    return dydt

def evaluate_pid_performance(Kp, Ki, Kd, setpoint=35.0, room_temp=16.8, plot=False):
    """计算给定PID参数下的系统性能指标
        Args:
            Kp/Ki/Kd: PID参数
            setpoint: 目标温度(默认35°C)
            room_temp: 环境温度(默认16.8°C)
            plot: 是否绘制响应曲线
        Returns:
            dict: 包含ISE/IAE/超调量等指标
        """
    # 仿真参数
    dt = 0.5
    t = np.arange(0, 20000.5,dt)

    # 初始化系统
    y0 = 0  # 系统初始输出(注意需叠加room_temp)
    Y = [y0]
    U = []
    errors = []
    u_history = np.zeros_like(t)  # 记录历史控制量

    # PID控制器(带输出限幅0-100)
    prev_error = 0
    integral = 0

    for i in range(1, len(t)):  # 从第1个时间点开始
        current_time = t[i]  # 手动获取当前时间点
        # 反馈信号(含室温和干扰)
        current_temp = Y[-1] + room_temp + disturbance(current_time)

        # 计算误差
        error = setpoint - current_temp
        errors.append(error)

        # PID计算
        P = Kp * error
        integral += error * dt
        I = Ki * integral
        D = Kd * (error - prev_error) / dt
        prev_error = error


        # 输出限幅
        u = np.clip(P + I + D, 0, 100)
        U.append(u)
        # 记录控制量
        u_history[i] = u
        # 系统响应
        y = odeint(system, Y[-1], [current_time, current_time + dt],
                   args=(u_history[:i + 1], t[:i + 1]))[-1, 0]
        Y.append(y)

    # 性能指标计算
    Y = np.array(Y) + room_temp
    errors = np.array(errors)

    # 调节时间
    # 查找调节时间（进入 ±2% 后不再超出）
    setpoint_tolerance = 0.02 * setpoint
    settling_index = None

    for i in range(len(Y)):
        # 检查从当前点到最后是否所有点都在容差范围内
        if all(abs(y - setpoint) <= setpoint_tolerance for y in Y[i:]):
            settling_index = i
            break

    metrics = {
        'state_error':Y[-1]-setpoint,# 稳态误差
        'ISE': np.sum(errors ** 2),  # 平方误差积分
        'IAE': np.sum(np.abs(errors)),  # 绝对误差积分
        'ITAE': np.sum(t[1:] * np.abs(errors)),  # 时间加权绝对误差
        'overshoot': max(0, np.max(Y) - setpoint),  # 超调量
        'settling_time': t[settling_index] if settling_index is not None else t[-1]+50000,  # 调节时间
        'control_effort': np.sum(np.abs(np.diff(U)))  # 控制量变化率
    }

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.plot(t, Y, label='Temperature')
        plt.axhline(setpoint, color='r', linestyle='--', label='Setpoint')
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid()
        plt.show()

    return metrics

if __name__ == '__main__':
    #进化算法 16次
    # Kp = 15.141524456102935
    # Ki = 0.07476709702086162
    # Kd = 1833.633658743375


    #pso 第四次迭代
    Kp = 0.3954348252555275
    Ki = 0.0
    Kd = 601.9046216460526

    metrics = evaluate_pid_performance(Kp, Ki, Kd,plot=True)
    print(metrics)