import numpy as np
import matplotlib.pyplot as plt
import math

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 故障时间数据
NTDSDataset = [9, 21, 32, 36, 43, 45, 50, 58, 63, 70, 71, 77, 78, 87, 91, 92, 95, 98, 104, 105,
     116, 149, 156, 247, 249, 250, 337, 384, 396, 405, 540, 798, 814, 849]

# 训练数据和验证数据
train_t = NTDSDataset[:26]
test_t = NTDSDataset[26:]


# 估计参数a和b
def GO_model(train_t):
    epsilon = 0.000_001
    N = len(train_t)
    D = sum(train_t) / (N * max(train_t))
    xl = xm = xr = 0
    if 0 < D < 1 / 2:
        xl = (1 - 2 * D) / 1
        xr = 1 / D
    else:
        raise ValueError("无解！")
    while 1:
        xm = (xl + xr) / 2
        if abs(xl - xr) < 1e-10:
            xm = (xl - xr) / 2
            break
        f = (1 - D * xm) * math.e ** xm + (D - 1) * xm - 1
        if f > epsilon:
            xl = xm
        elif f < -epsilon:
            xr = xm
        else:
            break

    b = xm / max(train_t)
    a = N / (1 - math.e ** (-b * max(train_t)))

    return a, b


a, b = GO_model(train_t)
print("a =", a, "b =", b)

# 计算累计故障数量
def m(t, a, b):
    return a * (1 - np.exp(-b * np.array(t)))


# 反解GO模型的累计故障时间
def inverse_m(m_values, a, b):
    return -1 / b * np.log(1 - np.array(m_values) / a)


# 训练数据的预测值
train_m = m(train_t, a, b)

# 验证数据的预测值
test_m = m(test_t, a, b)

# 绘制图表
plt.figure(figsize=(10, 6))
n = len(NTDSDataset)
plt.plot(range(1, len(train_t) + 1), train_t, 'bo-', label='训练数据')
plt.plot(range(len(train_t) + 1, n + 1), test_t, 'ro-', label='验证数据')
plt.plot(range(1, n + 1), train_t + test_t, 'b-')
plt.plot(m(NTDSDataset, a, b), NTDSDataset, 'g--', label='模型预测')
plt.scatter(m(NTDSDataset, a, b), NTDSDataset, color='green')  # 添加实心点
plt.scatter(range(1, len(train_t) + 1), train_t, color='blue')
plt.scatter(range(len(train_t) + 1, len(NTDSDataset) + 1), test_t, color='red')
plt.plot(range(len(train_t) + 1, len(NTDSDataset) + 1), test_t, 'r-')  # 修改为红色线条
plt.xlabel('故障时间')
plt.ylabel('累计故障数量')
plt.title('GO模型验证与可视化')
plt.legend()
plt.grid(True)
plt.show()

# plt.plot(train_t, range(1, len(train_t) + 1), 'bo-', label='训练数据')
# plt.plot(test_t, range(len(train_t) + 1, len(NTDSDataset) + 1), 'ro-', label='验证数据')
# plt.plot(train_t + test_t, range(1, len(NTDSDataset) + 1), 'b-')
# plt.plot(NTDSDataset, m(NTDSDataset, a, b), 'g--', label='模型预测')
# plt.scatter(NTDSDataset, m(NTDSDataset, a, b), color='green')  # 添加实心点
# plt.scatter(train_t, range(1, len(train_t) + 1), color='blue')
# plt.scatter(test_t, range(len(train_t) + 1, len(NTDSDataset) + 1), color='red')
# plt.plot(test_t, range(len(train_t) + 1, len(NTDSDataset) + 1), 'r-')  # 修改为红色线条
# plt.xlabel('故障时间')
# plt.ylabel('累计故障数量')
# plt.title('GO模型验证与可视化')
# plt.legend()
# plt.grid(True)
# plt.show()
