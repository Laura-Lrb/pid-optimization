import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from evaluate_pid import  evaluate_pid_performance

def optimize_pid_pso(initial_params, objective_func, bounds=None, max_iter=50):
    """PID参数优化(pso)
    Args:
        initial_params: 初始参数[Kp, Ki, Kd]
        objective_func: 目标函数(接收PID参数，返回标量值)
        bounds: 参数范围[(Kp_min,Kp_max), (Ki_min,Ki_max), (Kd_min,Kd_max)]
        max_iter: 最大迭代次数
    Returns:
        dict: 最佳参数和性能
    """
    population_log = []
    # 默认参数范围
    if bounds is None:
        bounds = [(100, 1000), (0.01, 1), (100, 5000)]

    # 初始化种群
    population_size = 20
    population = np.column_stack([
        np.random.uniform(low, high, population_size)
        for (low, high) in bounds
    ])
    speed = np.zeros((population_size, 3))
    # print( population)
    best_params = initial_params
    best_score = objective_func(*initial_params)['ISE']  # 以ISE为目标
    individual_best = population
    individual_best_scores=np.zeros(population_size)
    for i in range(population_size):
        result_i = objective_func(*population[i])
        individual_best_scores[i] = result_i['ISE']+ 0.2 * result_i['overshoot'] ** 2 + 0.2 * result_i['settling_time']
    # print(individual_best_scores)
    best_list = []

    for _ in range(max_iter):
        # 评估当前种群
        scores=np.zeros(population_size)
        for i in range(population_size):
            result_i = objective_func(*population[i])
            scores[i] = result_i['ISE'] + 0.2 * result_i['overshoot'] ** 2 + 0.2 * result_i['settling_time']
        print(scores)
        # 选择全局最优解
        min_idx = np.argmin(scores)
        if scores[min_idx] < best_score:
            best_params = population[min_idx]
            best_score = scores[min_idx]

        best_list.append({
            'iteration': _,
            'best_params': best_params.tolist(),  # 转为 list 方便保存
            'best_score': float(best_score),  # 转为 float
            'Kp': float(best_params[0]),
            'Ki': float(best_params[1]),
            'Kd': float(best_params[2]),
            'metrics': objective_func(*best_params)  # 获取完整指标
        })
        print(f"{_}次",best_list[-1])

        # 选择个体最优解
        for i in range(population_size):
            if scores[i] < individual_best_scores[i]:
                individual_best[i] = population[i]
                individual_best_scores[i] = scores[i]
            # 计算速度
            r = np.random.uniform(0, 1, 2)  # 生成两个随机数组
            speed_for_individual_best = r[0] * (individual_best[i] - population[i])
            speed_for_global_best = r[1] * (best_params - population[i])

            # 更新速度公式
            speed[i] = 0.4 * speed[i] + 2 * (speed_for_individual_best + speed_for_global_best)
            population[i] = population[i] + speed[i]

            # 限幅确保参数合法
            population[i] = np.clip(population[i], [b[0] for b in bounds], [b[1] for b in bounds])
            # print(f"{_}次{i}个个体",population[i])

        row_data = {}
        for idx, (p, s) in enumerate(zip(population, scores)):
            row_data[f'p{idx + 1}'] = p.tolist()  # 参数转为 list 以便保存
            row_data[f'score{idx + 1}'] = float(s)  # 分数转为 float

        # 添加迭代次数字段（可选）
        row_data['iteration'] = _

        # 追加到日志中
        population_log.append(row_data)

    log_df = pd.DataFrame(best_list)
    log_df.to_csv('pid_pso_log.csv', index=False)
    print("日志已保存至 pid_pso_log.csv")
    plog_df = pd.DataFrame([population_log])
    plog_df.to_csv('population_pso_log.csv', index=False)
    print("日志已保存至 population_pso_log.csv")

    return {
        'Kp': best_params[0],
        'Ki': best_params[1],
        'Kd': best_params[2],
        'performance': objective_func(*best_params)
    }

# def optimize_pid_pso(initial_params, objective_func, bounds=None, max_iter=50):

if __name__ == '__main__':
    # 优化方法
    result = optimize_pid_pso(
        initial_params=[300, 0.05, 500],
        objective_func=evaluate_pid_performance,
        bounds=[(0, 500), (0, 0.5), (0, 2000)]
    )