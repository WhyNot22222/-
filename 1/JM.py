import numpy as np
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 数据集
NTDSDataset = np.array([
    9, 12, 11, 4, 7, 2, 5, 8, 5, 7, 1, 6, 1, 9, 4, 1, 3, 3, 6, 1, 11, 33, 7, 91, 2, 1, 87, 47, 12, 9, 135, 258, 16, 35
])
NTDSDataset_old = np.array([
    9, 12, 11, 4, 7, 2, 5, 8, 5, 7, 1, 6, 1, 9, 4, 1, 3, 8, 6, 1, 1, 33, 7, 91, 2, 1, 87, 47, 12, 9, 135, 258, 16, 35
])


# 定义方程组
def equations(p, x):
    N0, phi = p
    n = len(x)
    f1 = n / (N0 * np.sum(x) - np.sum((np.arange(n)) * x)) - phi
    f2 = np.sum(1 / (N0 - np.arange(n))) - n / (N0 - (1 / np.sum(x)) * np.sum((np.arange(n)) * x))
    return np.array([f1, f2])

# 目标函数：方程组的平方和
def objective(p, x):
    return np.sum(equations(p, x) ** 2)  # 这里可以安全地做平方运算

# 参数估计
def estimate_parameters(x):
    n = len(x)
    initial_guess = [32, 0.01]  # 初始猜测值
    result = basinhopping(objective, initial_guess, niter_success=50,
                          minimizer_kwargs={"method": "BFGS", "args": (x,)})
    return result.x

# 计算累计故障时间
def cumulative_failure_time(data):
    return np.cumsum(data)

# 计算预测的累计故障时间
def predict_cumulative_failure_time(N0, phi, data_length):
    failure_times = []
    current_time = 0
    for i in range(1, data_length + 1):
        current_time += 1 / (phi * (N0 - i + 1))
        failure_times.append(current_time)
    return failure_times

# 主函数
def main():
    # 估计参数
    # x = NTDSDataset[:26]
    # x = NTDSDataset_old[:26]
    x = NTDSDataset
    N0, phi = estimate_parameters(x)
    print(f"估计的N0: {N0}, 估计的phi: {phi}")

    # 计算实际的累计故障时间
    actual_cumulative_time = cumulative_failure_time(NTDSDataset)

    # 计算预测的累计故障时间，绘制到N0为止
    N = int(N0) + 1
    predicted_cumulative_time = predict_cumulative_failure_time(N0, phi, N)

    # 绘制实际和预测的累计故障时间
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(NTDSDataset) + 1), actual_cumulative_time, label='实际累计故障时间', marker='o')
    plt.plot(range(1, N + 1), predicted_cumulative_time[:N], label='预测累计故障时间', marker='x')
    plt.xlabel('故障次数')
    plt.ylabel('累计故障时间')
    plt.title('实际与预测累计故障时间比较')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
