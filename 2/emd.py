import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

# 假设 data 是你的NTDS数据集
data = np.array([
    9, 12, 11, 4, 7, 2, 5, 8, 5, 7, 1, 6, 1, 9, 4, 1, 3, 3, 6, 1, 11, 33, 7, 91, 2, 1, 87, 47, 12, 9, 135, 258, 16, 35
])

# 标准化数据
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

# EMD分解
emd = EMD()
imfs = emd.emd(data_scaled)
residue = data_scaled - np.sum(imfs, axis=0)

# 绘制IMFs和残差
plt.figure(figsize=(12, 8))
for i, imf in enumerate(imfs):
    plt.subplot(len(imfs) + 1, 1, i + 1)
    plt.plot(imf)
    plt.title(f'IMF {i+1}')
plt.subplot(len(imfs) + 1, 1, len(imfs) + 1)
plt.plot(residue)
plt.title('Residue')
plt.tight_layout()
plt.show()

# 对每个IMF进行随机森林建模
rf_predictions = []
for imf in imfs:
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(np.arange(len(imf)).reshape(-1, 1), imf)
    rf_pred = rf.predict(np.arange(len(imf)).reshape(-1, 1))
    rf_predictions.append(rf_pred)

# 合并预测结果
combined_rf_prediction = np.sum(rf_predictions, axis=0)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(data, label='Original Data')
plt.plot(combined_rf_prediction, label='Combined RF Prediction', linestyle='--')
plt.legend()
plt.title('Improved RF Prediction vs Original Data')
plt.show()
