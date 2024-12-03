import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def difference(data, interval=1):
    return np.diff(data, n=interval)

def inverse_difference(original_data, diff_data, interval=1):
    return np.r_[original_data[:interval], diff_data].cumsum()

def ar_model(data, p):
    X = np.column_stack([data[i:len(data) - p + i] for i in range(p)])
    y = data[p:]
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return coef

def ma_model(errors, q):
    X = np.column_stack([errors[i:len(errors) - q + i] for i in range(q)])
    y = errors[q:]
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return coef

def arima_fit(data, p, d, q):
    diff_data = difference(data, d)
    
    # AR模型
    ar_coef = ar_model(diff_data, p)
    
    # MA模型
    errors = diff_data[p:] - np.dot(np.column_stack([diff_data[i:len(diff_data) - p + i] for i in range(p)]), ar_coef)
    ma_coef = ma_model(errors, q)
    
    return ar_coef, ma_coef

def arima_forecast(data, ar_coef, ma_coef, p, q):
    diff_data = difference(data, 1)
    forecast = []
    
    for i in range(len(diff_data) - p):
        ar_part = np.dot(diff_data[i:i+p], ar_coef)
        ma_part = np.dot(diff_data[i:i+q], ma_coef)
        forecast.append(ar_part + ma_part)
    
    forecast = inverse_difference(data, np.array(forecast), 1)
    forecast = np.abs(forecast)
    return forecast

if __name__ == "__main__":
    NTDSDataset = np.array([
        9, 12, 11, 4, 7, 2, 5, 8, 5, 7, 1, 6, 1, 9, 4, 1, 3, 3, 6, 1, 11, 33, 7, 91, 2, 1, 87, 47, 12, 9, 135, 258, 16, 35
    ])
    p, d, q = 2, 1, 2
    ar_coef, ma_coef = arima_fit(NTDSDataset, p, d, q)
    forecast = arima_forecast(NTDSDataset, ar_coef, ma_coef, p, q)
    
    plt.plot(range(1, len(NTDSDataset) + 1), NTDSDataset, label='原始数据', marker='o')
    plt.plot(range(1, len(forecast) + 1), forecast, label='ARIMA预测', marker='x')
    plt.title('ARIMA模型预测')
    plt.xlabel('时间')
    plt.ylabel('值')
    plt.legend()
    plt.show()
