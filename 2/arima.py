from statsmodels.tsa.arima.model import ARIMA  # 导入ARIMA模型
import numpy as np
import matplotlib.pyplot as plt  # 导入可视化库

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def arima_fit(data, p, d, q):
    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit()
    return model_fit  # 返回模型拟合对象

if __name__ == "__main__":
    # 累计故障数量
    cumulative_failures = np.arange(1, 33)  # 从1到32的数组
    # 累计失效时间
    cumulative_time = np.array([
        9, 12, 11, 4, 7, 2, 5, 8, 5, 7, 1, 6, 1, 9, 4, 1, 3, 3, 6, 1, 11, 33, 7, 91, 2, 1, 87, 47, 12, 9, 135, 258, 16, 35
    ])
    
    # 确保累计故障数量和累计失效时间的长度一致
    if len(cumulative_failures) != len(cumulative_time):
        min_length = min(len(cumulative_failures), len(cumulative_time))
        cumulative_failures = cumulative_failures[:min_length]
        cumulative_time = cumulative_time[:min_length]

    p, d, q = 2, 1, 2
    model_fit = arima_fit(cumulative_time, p, d, q)  # 获取模型拟合对象
    fitted_values = model_fit.fittedvalues  # 获取拟合值

    # 可视化结果
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_failures, cumulative_time, label='实际数据', marker='o')
    plt.plot(cumulative_failures[:-1], fitted_values[1:], label='拟合数据', marker='x')  # 向右移动一个点
    plt.title('ARIMA 模型拟合')
    plt.xlabel('累计故障数量')
    plt.ylabel('累计失效时间')
    plt.legend()
    plt.show()  # 显示图形