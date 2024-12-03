import numpy as np
import matplotlib.pyplot as plt
from JM import estimate_parameters, cumulative_failure_time, predict_cumulative_failure_time
from GO import inverse_m, GO_model

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 使用相同的数据集
NTDSDataset = np.array([
    9, 12, 11, 4, 7, 2, 5, 8, 5, 7, 1, 6, 1, 9, 4, 1, 3, 3, 6, 1, 11, 33, 7, 91, 2, 1, 87, 47, 12, 9, 135, 258, 16, 35
])

# JM模型预测
N0, phi = estimate_parameters(NTDSDataset[:26])
print(f"JM模型：N0 = {N0}, phi = {phi}")
actual_cumulative_time = cumulative_failure_time(NTDSDataset)
predicted_cumulative_time_jm = predict_cumulative_failure_time(N0, phi, len(NTDSDataset))

# GO模型预测
train_t = np.cumsum(NTDSDataset[:26])
a, b = GO_model(train_t)
print(f"GO模型：a = {a}, b = {b}")
predicted_cumulative_time_go = inverse_m(range(1, len(NTDSDataset) + 1), a, b)

# 绘制比较图
N = int(N0) + 1
plt.figure(figsize=(12, 8))
plt.plot(range(1, len(NTDSDataset) + 1), actual_cumulative_time, label='实际累计故障时间', marker='o')
plt.plot(range(1, N + 1), predicted_cumulative_time_jm[:N], label='JM模型预测', marker='x')
plt.plot(range(1, len(NTDSDataset) + 1), predicted_cumulative_time_go, label='GO模型预测', marker='s')
plt.xlabel('故障次数')
plt.ylabel('累计故障时间')
plt.title('JM模型与GO模型的比较')
plt.legend()
plt.grid(True)
plt.show()
