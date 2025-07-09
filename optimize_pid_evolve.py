import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from evaluate_pid import  evaluate_pid_performance

def optimize_pid_evolve(initial_params, objective_func, bounds=None, max_iter=50):
    """PID参数优化(遗传算法)
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

    best_params = initial_params
    best_score = objective_func(*initial_params)['ISE']  # 以ISE为目标

    best_list = []

    for _ in range(max_iter):
        # 评估当前种群
        # scores = []
        scores = np.array([
            objective_func(*params)['ISE'] + 0.2 * objective_func(*params)['overshoot'] ** 2 + 0.1 * objective_func(*params)['settling_time']
            for params in population
        ])
        print(scores)

        # 选择最优个体
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


        # 轮盘赌概率
        probabilities = np.exp(-scores / np.sum(scores))
        probabilities /= probabilities.sum()

        # 生成新一代
        new_population = []
        for _ in range(population_size):
            # 轮盘赌选择
            parent_indices = np.random.choice(
                a=len(population),  # 种群大小
                size=2,  # 选择两个个体
                replace=False,  # 不放回（确保不同）
                p=probabilities  # 每个个体被选中的概率
            )

            parent1 = population[parent_indices[0]]
            parent2 = population[parent_indices[1]]

            # 单点交叉
            crossover_point = np.random.randint(1, len(parent1))  # 随机选择交叉点
            child = np.empty_like(parent1)
            child[:crossover_point] = parent1[:crossover_point]
            child[crossover_point:] = parent2[crossover_point:]

            # 随机高斯变异
            child *= np.random.normal(1, 0.1, size=3)

            # 确保参数在范围内
            child = np.clip(child, [b[0] for b in bounds], [b[1] for b in bounds])
            new_population.append(child)

        row_data = {}
        for idx, (p, s) in enumerate(zip(population, scores)):
            row_data[f'p{idx + 1}'] = p.tolist()  # 参数转为 list 以便保存
            row_data[f'score{idx + 1}'] = float(s)  # 分数转为 float

        # 添加迭代次数字段（可选）
        row_data['iteration'] = _

        # 追加到日志中
        population_log.append(row_data)

        population = np.array(new_population)

    log_df = pd.DataFrame(best_list)
    log_df.to_csv('pid_evolution_log.csv', index=False)
    print("日志已保存至 pid_evolution_log.csv")
    plog_df = pd.DataFrame([population_log])
    plog_df.to_csv('population_log.csv', index=False)
    print("日志已保存至 population_log.csv")

    return {
        'Kp': best_params[0],
        'Ki': best_params[1],
        'Kd': best_params[2],
        'performance': objective_func(*best_params)
    }

# def optimize_pid_pso(initial_params, objective_func, bounds=None, max_iter=50):

if __name__ == '__main__':
    # 优化方法
    result = optimize_pid_evolve(
        initial_params=[300, 0.05, 500],
        objective_func=evaluate_pid_performance,
        bounds=[(0, 500), (0, 0.5), (0, 2000)]
    )
    # print(f"Optimized PID: Kp={result['Kp']:.2f}, Ki={result['Ki']:.4f}, Kd={result['Kd']:.2f}")
    # print("Optimized performance:", result['performance'])
    # 测试
    # Kp = 100.00
    # Ki = 0.0560
    # Kd = 1318.81
    # metrics = evaluate_pid_performance(Kp, Ki, Kd,plot=True)
    # print(metrics)