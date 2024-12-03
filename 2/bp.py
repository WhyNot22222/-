import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

origin_data = [
    500, 800, 1000, 1100, 1210, 1320, 1390, 1500, 1630, 1700,
    1890, 1960, 2010, 2100, 2150, 2230, 2350, 2470, 2500, 3000,
    3050, 3110, 3170, 3230, 3290, 3320, 3350, 3430, 3480, 3495,
    3560, 3720, 3750, 3780, 3810, 3830, 3855, 3876, 3896, 3908,
    3920, 3950, 3975, 3982
]

data = origin_data

# 实例化模型、损失函数和优化器
hidden_size = 20
output_size = 1
learning_rate = 1e-3
num_epochs = 100000


# 准备训练数据
def create_dataset(data, input_size):
    X, y = [], []
    for i in range(len(data) - input_size):
        X.append(data[i:i + input_size])
        y.append(data[i + input_size])
    return np.array(X), np.array(y)


input_size = 5
X_train, y_train = create_dataset(data[:40], input_size)
X_test, y_test = create_dataset(data[35:], input_size)

# 转换为张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# 定义BP神经网络模型
class BPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BPModel, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size * 2)
        self.relu = nn.ReLU()
        self.hidden2 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.hidden3 = nn.Linear(hidden_size * 2, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.hidden3(x)
        x = self.relu(x)
        x = self.output(x)
        return x


model = BPModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
losses = []
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型并可视化预测结果
model.eval()
with torch.no_grad():
    test_outputs = model(X_test).numpy()

y_test = y_test.numpy().flatten().tolist()
test_outputs = test_outputs.flatten().tolist()
print('actual:', y_test)
print('test_outputs:', test_outputs)

# 绘制预测结果与实际值
plt.figure(figsize=(10, 5))

# 绘制实际值
plt.plot(y_test, label='Actual Values', color='blue', marker='o', linestyle='-')

# 绘制预测值
plt.plot(test_outputs, label='Predicted Values', color='orange', marker='x', linestyle='--')

plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.grid(True)  # 增加网格线
plt.show()
